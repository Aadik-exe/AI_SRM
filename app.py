import asyncio
import math
import uuid
import time
import base64
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.enums import Resampling
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ============================================================================
# CONFIG & LOGGING
# ============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
DEMO_DIR = DATA_DIR / "demo"
MODELS_DIR = BASE_DIR / "models"
for d in [INPUT_DIR, OUTPUT_DIR, DEMO_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_PX = 192   # SR output will be 3× this = 576px — fast on CPU/MPS
_EXECUTOR = ThreadPoolExecutor(max_workers=2)

# ============================================================================
# TIFF / IMAGE LOADING
# ============================================================================
def load_image_any(path: str) -> tuple[np.ndarray, dict]:
    """
    Load any image (TIFF, PNG, JPG) into a uint8 RGB numpy array.
    Returns (rgb_image, metadata_dict).
    For multi-band GeoTIFF: reads bands 1-3 (R,G,B), normalises to uint8.
    For single-band: converts to 3-channel greyscale.
    """
    meta = {"source": "unknown", "bands": 1, "crs": None, "transform": None, "dtype": "uint8"}

    try:
        with rasterio.open(path) as src:
            meta["bands"] = src.count
            meta["crs"] = str(src.crs) if src.crs else None
            meta["transform"] = str(src.transform) if src.transform else None
            meta["width"] = src.width
            meta["height"] = src.height
            meta["dtype"] = str(src.dtypes[0])
            meta["source"] = "rasterio"

            # Choose first 3 bands (or fewer)
            band_count = min(3, src.count)
            arrays = []
            for b in range(1, band_count + 1):
                band = src.read(b).astype(np.float32)
                # Normalise to [0,255]
                lo, hi = np.percentile(band, 2), np.percentile(band, 98)
                if hi > lo:
                    band = np.clip((band - lo) / (hi - lo) * 255, 0, 255)
                else:
                    band = np.zeros_like(band)
                arrays.append(band.astype(np.uint8))

            if band_count == 1:
                img_rgb = np.stack([arrays[0], arrays[0], arrays[0]], axis=2)
            elif band_count == 2:
                img_rgb = np.stack([arrays[0], arrays[1], arrays[0]], axis=2)
            else:
                img_rgb = np.stack(arrays[:3], axis=2)

    except Exception as e:
        logger.warning(f"rasterio failed ({e}), falling back to OpenCV")
        img_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {path}")
        if img_bgr.ndim == 2:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        elif img_bgr.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        meta["source"] = "opencv"
        meta["bands"] = 1
        meta["width"] = img_rgb.shape[1]
        meta["height"] = img_rgb.shape[0]

    # Cap resolution
    h, w = img_rgb.shape[:2]
    if max(h, w) > MAX_IMAGE_PX:
        scale = MAX_IMAGE_PX / max(h, w)
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized from {w}x{h} to {img_rgb.shape[1]}x{img_rgb.shape[0]}")

    return img_rgb, meta


def numpy_to_base64(img_array: np.ndarray) -> str:
    if img_array.ndim == 2:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.png', img_bgr)
    return f"data:image/png;base64,{base64.b64encode(buf).decode()}"


def generate_demo_image(output_path: str, width: int = 256, height: int = 256) -> str:
    rng = np.random.default_rng(42)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = [52, 92, 45]
    field_colors = [[42, 82, 38], [58, 105, 52], [35, 70, 32], [65, 110, 58], [48, 88, 42]]
    for i, sy in enumerate(range(0, height, 32)):
        img[sy:sy+32, :] = field_colors[i % len(field_colors)]
    for y in range(0, height, 32): img[y:y+1, :] = [30, 55, 28]
    for x in range(0, width, 30): img[:, x:x+1] = [30, 55, 28]
    blocks = [(10,10,75,55),(85,10,145,45),(10,65,55,110),(65,65,130,105),
              (145,10,200,65),(140,75,210,130),(10,125,100,165),(110,125,215,170),
              (10,175,130,215),(145,185,215,225)]
    for x1,y1,x2,y2 in blocks:
        bg = int(rng.integers(105, 135))
        bc = np.array([bg, bg-3, bg+5])
        img[y1:y2, x1:x2] = np.clip(bc, 80, 165)
        for by in range(y1+4, y2-4, 7):
            for bx in range(x1+4, x2-4, 7):
                img[by:by+4, bx:bx+4] = np.clip(bc-20, 60, 140)
        cv2.rectangle(img, (x1,y1), (x2,y2), (max(0,bg-20), max(0,bg-22), max(0,bg-18)), 1)
    img[92:100, :] = [158, 154, 148]; img[:, 138:146] = [158, 154, 148]
    img[48:54, 10:138] = [150, 147, 142]; img[10:92, 72:78] = [150, 147, 142]
    img[105:112, 145:256] = [150, 147, 142]
    cv2.fillPoly(img, [np.array([[215,15],[248,15],[252,80],[220,85],[210,50]], np.int32)], [55, 108, 168])
    for cx, cy, r in [(230,130,22),(245,175,15),(220,210,18),(235,240,12)]:
        cv2.circle(img, (cx, cy), r, [38,88,35], -1)
        cv2.circle(img, (cx-3, cy-3), r//2, [55,110,48], -1)
    noise = rng.integers(-6, 6, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.9)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return output_path


# ============================================================================
# QUALITY METRICS
# ============================================================================
def compute_psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    mse = np.mean((original.astype(np.float32) - enhanced.astype(np.float32)) ** 2)
    if mse == 0: return 100.0
    return round(float(20 * math.log10(255.0 / math.sqrt(mse))), 2)

def compute_ssim(original: np.ndarray, enhanced: np.ndarray) -> float:
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    o, e = original.astype(np.float64), enhanced.astype(np.float64)
    mu1, mu2 = cv2.GaussianBlur(o,(11,11),1.5), cv2.GaussianBlur(e,(11,11),1.5)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1*mu2
    s1 = cv2.GaussianBlur(o**2,(11,11),1.5)-mu1_sq
    s2 = cv2.GaussianBlur(e**2,(11,11),1.5)-mu2_sq
    s12 = cv2.GaussianBlur(o*e,(11,11),1.5)-mu12
    m = ((2*mu12+C1)*(2*s12+C2))/((mu1_sq+mu2_sq+C1)*(s1+s2+C2))
    return round(float(m.mean()), 4)

def compute_sam(original: np.ndarray, enhanced: np.ndarray) -> float:
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    o, e = original.astype(np.float32), enhanced.astype(np.float32)
    dot = np.sum(o*e, axis=2)
    denom = np.linalg.norm(o,axis=2)*np.linalg.norm(e,axis=2)
    denom[denom==0] = 1e-8
    return round(float(np.mean(np.arccos(np.clip(dot/denom,-1,1)))*(180/math.pi)), 4)

def compute_spectral_consistency(original: np.ndarray, enhanced: np.ndarray) -> float:
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    ch = original.shape[2] if original.ndim==3 else 1
    if ch == 1:
        original, enhanced = original[:,:,np.newaxis], enhanced[:,:,np.newaxis]
    bcs = []
    for c in range(ch):
        h1 = cv2.calcHist([original[:,:,c]],[0],None,[64],[0,256]).flatten()
        h2 = cv2.calcHist([enhanced[:,:,c]],[0],None,[64],[0,256]).flatten()
        h1 /= h1.sum()+1e-8; h2 /= h2.sum()+1e-8
        bcs.append(np.clip(float(np.sum(np.sqrt(h1*h2+1e-12))),0,1))
    return round(float(np.mean(bcs))*100, 1)

def compute_sharpness_gain(original: np.ndarray, enhanced: np.ndarray) -> float:
    if original.ndim==3: original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    if enhanced.ndim==3: enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    ol = cv2.Laplacian(original, cv2.CV_64F).var()
    el = cv2.Laplacian(enhanced, cv2.CV_64F).var()
    if ol < 1e-5: return 1.0
    return round(float(el/ol), 2)


# ============================================================================
# GEOSPATIAL ANALYZER
# ============================================================================
class GeospatialAnalyzer:
    @staticmethod
    def extract_features(rgb: np.ndarray) -> dict:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Urban / Buildings (Canny)
        edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0), 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Improved: Removed convexity check. Use area and perimeter to find complex structures
        bldgs = [c for c in cnts if 10 < cv2.contourArea(c) and cv2.arcLength(c, True) < 1500]
        bm = np.zeros_like(gray); cv2.drawContours(bm, bldgs, -1, 255, -1)
        em = np.zeros_like(gray); cv2.drawContours(em, bldgs, -1, 255, 2)
        
        r,g,b = cv2.split(rgb.astype(np.float32))
        
        # 2. Agriculture / Vegetation (VARI: Visible Atmospherically Resistant Index)
        # Formula: (G - R) / (G + R - B) -> Scientifically accurate for RGB satellite imagery
        denom = g + r - b
        denom[denom == 0] = 1e-5
        vari = (g - r) / denom
        
        vm = cv2.morphologyEx(np.where(vari > 0.05, 255, 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        
        # 2a. Crop Health
        healthy_mask = np.zeros_like(gray)
        stressed_mask = np.zeros_like(gray)
        if np.sum(vm) > 0:
            healthy_mask = np.where((vm > 0) & (vari > 0.15), 255, 0).astype(np.uint8)
            stressed_mask = np.where((vm > 0) & (vari <= 0.15), 255, 0).astype(np.uint8)

        # 3. Hydrology / Water
        # Refined heuristic: Water is dark overall, Blue dominant over Red
        water_mask = np.where((b > r + 5) & (g > r) & (r < 90) & (g < 110) & (b < 120), 255, 0).astype(np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        
        total_pixels = gray.size
        
        return {
            "buildup_percent": round(float(np.sum(bm>0)/total_pixels*100),1),
            "vegetation_percent": round(float(np.sum(vm>0)/total_pixels*100),1),
            "healthy_crop_percent": round(float(np.sum(healthy_mask>0)/total_pixels*100),1),
            "stressed_crop_percent": round(float(np.sum(stressed_mask>0)/total_pixels*100),1),
            "water_percent": round(float(np.sum(water_mask>0)/total_pixels*100),1),
            "detected_structures": len(bldgs),
            "building_mask": bm, "veg_mask": vm, "edge_mask": em,
            "healthy_mask": healthy_mask, "stressed_mask": stressed_mask, "water_mask": water_mask
        }


# ============================================================================
# SUPER RESOLUTION MODEL (EDSR-Lite)
# ============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, f=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(f,f,3,padding=1), nn.ReLU(True), nn.Conv2d(f,f,3,padding=1))
    def forward(self, x): return x + self.net(x)

class EDSR(nn.Module):
    def __init__(self, nc=3, nf=32, nb=8, scale=3):
        super().__init__()
        self.head = nn.Conv2d(nc, nf, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])
        self.tail = nn.Sequential(nn.Conv2d(nf, nf*(scale**2), 3, padding=1), nn.PixelShuffle(scale), nn.Conv2d(nf, nc, 3, padding=1))
    def forward(self, x):
        h = self.head(x); return self.tail(self.body(h)+h)

class SuperResolutionModel:
    def __init__(self):
        # Prefer Apple Metal (MPS) → CUDA → CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.scale_factor = 3
        self.model = EDSR(nc=3, nf=32, nb=8, scale=self.scale_factor).to(self.device)
        weight_path = MODELS_DIR / "edsr_lite.pth"
        if weight_path.exists():
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            self.model_name = "EDSR-Lite (Pretrained)"
        else:
            self.model_name = "EDSR-Lite (Fast Algorithmic)"
        self.model.eval()
        logger.info(f"SR device: {self.device}")

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).float().div(255).permute(2,0,1).unsqueeze(0).to(self.device)

    def _from_tensor(self, t: torch.Tensor) -> np.ndarray:
        return (t.squeeze(0).permute(1,2,0).cpu().clamp(0,1).numpy()*255).astype(np.uint8)

    def _bicubic_enhance(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        # Use LANCZOS4 for scientifically better high-frequency preservation than CUBIC
        up = cv2.resize(img, (w*self.scale_factor, h*self.scale_factor), interpolation=cv2.INTER_LANCZOS4)
        
        # detailEnhance provides much cleaner edge sharpening than unsharp mask
        sharp = cv2.detailEnhance(up, sigma_s=10, sigma_r=0.15)
        
        # Gentler CLAHE to avoid "deep fried" colors
        lab = cv2.cvtColor(sharp, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _tile_infer(self, img: np.ndarray, tile=128, overlap=16) -> np.ndarray:
        """Tiled inference to handle large images on CPU."""
        h, w = img.shape[:2]
        out_h, out_w = h * self.scale_factor, w * self.scale_factor
        output = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight = np.zeros((out_h, out_w), dtype=np.float32)

        step = tile - overlap
        for y in range(0, h, step):
            for x in range(0, w, step):
                y1, x1 = y, x
                y2, x2 = min(y+tile, h), min(x+tile, w)
                patch = img[y1:y2, x1:x2]
                with torch.no_grad():
                    t = self._to_tensor(patch)
                    sr_t = self.model(t)
                    sr_patch = self._from_tensor(sr_t).astype(np.float32)

                oy1, ox1 = y1*self.scale_factor, x1*self.scale_factor
                oy2, ox2 = y2*self.scale_factor, x2*self.scale_factor
                output[oy1:oy2, ox1:ox2] += sr_patch[:oy2-oy1,:ox2-ox1]
                weight[oy1:oy2, ox1:ox2] += 1.0

        weight = np.maximum(weight[:,:,np.newaxis], 1)
        return np.clip(output / weight, 0, 255).astype(np.uint8)

    def infer(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        has_pretrained = "Pretrained" in self.model_name

        if has_pretrained:
            sr_img = self._tile_infer(image_rgb)
        else:
            # Fast path: bicubic upscale + unsharp mask + CLAHE (no neural overhead)
            sr_img = self._bicubic_enhance(image_rgb)

        # Confidence map
        lr_up = cv2.resize(image_rgb, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        diff_gray = cv2.cvtColor(cv2.absdiff(lr_up, sr_img), cv2.COLOR_RGB2GRAY)
        conf_map = 255 - np.clip(diff_gray * 2, 0, 255)
        return sr_img, conf_map

sr_model = SuperResolutionModel()
logger.info(f"Model loaded: {sr_model.model_name}")

# ============================================================================
# PROCESSING HELPER
# ============================================================================
def run_full_pipeline(lr_img: np.ndarray, hr_ref: Optional[np.ndarray], meta: dict) -> dict:
    """Core pipeline: infer SR → metrics → geospatial → package response."""
    t0 = time.time()
    sr_img, conf_map = sr_model.infer(lr_img)
    proc_time = round(time.time()-t0, 2)

    # --- Metrics ---
    metrics = {
        "spectral_consistency": compute_spectral_consistency(lr_img, sr_img),
        "sharpness_gain": compute_sharpness_gain(lr_img, sr_img),
        "reference_available": hr_ref is not None,
        "benchmark_mode": hr_ref is not None,
        "psnr": None, "ssim": None, "sam": None,
    }
    if hr_ref is not None:
        metrics["psnr"] = compute_psnr(hr_ref, sr_img)
        metrics["ssim"] = compute_ssim(hr_ref, sr_img)
        metrics["sam"] = compute_sam(hr_ref, sr_img)
    else:
        baseline = cv2.resize(lr_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        metrics["psnr"] = compute_psnr(baseline, sr_img)
        metrics["ssim"] = compute_ssim(baseline, sr_img)
        metrics["sam"] = compute_sam(baseline, sr_img)

    # --- Geospatial ---
    analyzer = GeospatialAnalyzer()
    geo_lr = analyzer.extract_features(lr_img)
    geo_sr = analyzer.extract_features(sr_img)
    geo_stats = {
        "buildup_percent": geo_sr["buildup_percent"],
        "vegetation_percent": geo_sr["vegetation_percent"],
        "healthy_crop_percent": geo_sr["healthy_crop_percent"],
        "stressed_crop_percent": geo_sr["stressed_crop_percent"],
        "water_percent": geo_sr["water_percent"],
        "detected_structures": geo_sr["detected_structures"],
        "edge_improvement_factor": compute_sharpness_gain(geo_lr["edge_mask"], geo_sr["edge_mask"])
    }

    # --- Overlays ---
    conf_heat = cv2.applyColorMap(conf_map, cv2.COLORMAP_JET)
    overlay = sr_img.copy()
    # Water (Blue)
    overlay[geo_sr["water_mask"]>0] = np.array([30,100,250], dtype=np.uint8)
    # Healthy Crop (Dark Green)
    overlay[geo_sr["healthy_mask"]>0] = np.array([34,139,34], dtype=np.uint8)
    # Stressed Crop (Yellow-Green)
    overlay[geo_sr["stressed_mask"]>0] = np.array([200,220,50], dtype=np.uint8)
    # Buildings (Red) - drawn last so they overlay on top of anything else if overlaps happen
    overlay[geo_sr["building_mask"]>0] = np.array([255,80,80], dtype=np.uint8)
    
    overlay = cv2.addWeighted(sr_img, 0.65, overlay, 0.35, 0)

    return {
        "success": True,
        "processing_time": proc_time,
        "model_name": sr_model.model_name,
        "source_info": meta,
        "metrics": metrics,
        "geospatial": geo_stats,
        "images": {
            "input": numpy_to_base64(lr_img),
            "output": numpy_to_base64(sr_img),
            "confidence": numpy_to_base64(conf_heat),
            "geo_overlay": numpy_to_base64(overlay)
        },
        "dimensions": {
            "input": {"width": lr_img.shape[1], "height": lr_img.shape[0]},
            "output": {"width": sr_img.shape[1], "height": sr_img.shape[0]},
            "scale_factor": sr_model.scale_factor
        }
    }


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(title="Satellite AI – Super Resolution API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
def health():
    return {"status": "ok", "model": sr_model.model_name, "device": str(sr_model.device)}

@app.post("/api/process-demo")
async def process_demo():
    demo_path = str(DEMO_DIR / "sentinel2_demo.png")
    if not Path(demo_path).exists():
        generate_demo_image(demo_path)
    hr_ref, meta = load_image_any(demo_path)
    lr_img = cv2.resize(hr_ref, (hr_ref.shape[1]//3, hr_ref.shape[0]//3), interpolation=cv2.INTER_AREA)
    meta["mode"] = "demo"
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_EXECUTOR, run_full_pipeline, lr_img, hr_ref, meta)
    result["session_id"] = f"demo_{int(time.time())}"
    return JSONResponse(result)

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload any satellite image (GeoTIFF, PNG, JPG, TIFF).
    Reads real geospatial metadata from GeoTIFFs via rasterio.
    Runs the full SR + metrics + geospatial pipeline on it.
    """
    allowed_ext = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(400, detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_ext)}")

    # Save upload to disk
    session_id = str(uuid.uuid4())[:8]
    upload_path = INPUT_DIR / f"{session_id}{ext}"
    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Uploaded: {file.filename} → {upload_path} ({upload_path.stat().st_size/1024:.1f} KB)")
        lr_img, meta = load_image_any(str(upload_path))
        meta["original_filename"] = file.filename
        meta["session_id"] = session_id
        meta["mode"] = "upload"
        if min(lr_img.shape[:2]) < 16:
            raise HTTPException(400, detail="Image too small (minimum 16×16 pixels).")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_EXECUTOR, run_full_pipeline, lr_img, None, meta)
        result["session_id"] = session_id
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload processing failed")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    finally:
        # Clean up uploaded file
        if upload_path.exists():
            upload_path.unlink()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

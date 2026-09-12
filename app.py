"""
Satellite AI — Deep Learning Super Resolution Mapping
Problem Statement 26142 | NTRO | SIH 2024

Pipeline: Sentinel-2 (10 m) → Pre-processing → A2N ×4 → 2.5 m output
Applications: Crop Monitoring · Urban Analysis · Disaster Assessment
"""

import asyncio
import base64
import logging
import math
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rasterio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("srm")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
INPUT_DIR  = DATA_DIR / "input"
DEMO_DIR   = DATA_DIR / "demo"
MODELS_DIR = BASE_DIR / "models"
for d in [INPUT_DIR, DEMO_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_INPUT_PX  = 256   # max edge length of LR input (output = 256×4 = 1024 px)
SCALE_FACTOR  = 4     # 10 m Sentinel-2  →  2.5 m
_EXECUTOR     = ThreadPoolExecutor(max_workers=2)

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE I/O
# ══════════════════════════════════════════════════════════════════════════════
def load_image_any(path: str) -> tuple[np.ndarray, dict]:
    """Load any satellite image (GeoTIFF, PNG, JPG) into uint8 RGB + metadata."""
    meta: dict = {
        "source": "unknown", "bands": 1,
        "crs": None, "transform": None,
        "width": 0, "height": 0, "dtype": "uint8",
    }

    # Try rasterio (handles GeoTIFF natively) ----------------------------------
    try:
        with rasterio.open(path) as src:
            meta.update({
                "bands":     src.count,
                "crs":       str(src.crs) if src.crs else None,
                "transform": str(src.transform),
                "width":     src.width,
                "height":    src.height,
                "dtype":     str(src.dtypes[0]),
                "source":    "rasterio (GeoTIFF)",
            })
            n = min(3, src.count)
            arrays = []
            for b in range(1, n + 1):
                band = src.read(b).astype(np.float32)
                lo, hi = np.percentile(band, 2), np.percentile(band, 98)
                band = np.clip((band - lo) / (hi - lo + 1e-8) * 255, 0, 255)
                arrays.append(band.astype(np.uint8))
            if n == 1:
                img_rgb = np.stack([arrays[0]] * 3, axis=2)
            elif n == 2:
                img_rgb = np.stack([arrays[0], arrays[1], arrays[0]], axis=2)
            else:
                img_rgb = np.stack(arrays[:3], axis=2)
    except Exception as exc:
        logger.debug(f"rasterio failed ({exc}), falling back to OpenCV")
        img_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {path}")
        if img_bgr.ndim == 2:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        elif img_bgr.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        meta.update({
            "source": "OpenCV",
            "width":  img_rgb.shape[1],
            "height": img_rgb.shape[0],
        })

    # Cap resolution -----------------------------------------------------------
    h, w = img_rgb.shape[:2]
    if max(h, w) > MAX_INPUT_PX:
        s = MAX_INPUT_PX / max(h, w)
        img_rgb = cv2.resize(
            img_rgb,
            (int(w * s), int(h * s)),
            interpolation=cv2.INTER_AREA,
        )
        logger.info(f"Input capped {w}×{h} → {img_rgb.shape[1]}×{img_rgb.shape[0]}")

    return img_rgb, meta


def to_b64(arr: np.ndarray) -> str:
    bgr = (cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
           if arr.ndim == 2
           else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    _, buf = cv2.imencode(".png", bgr)
    return f"data:image/png;base64,{base64.b64encode(buf).decode()}"


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_psnr(ref: np.ndarray, enh: np.ndarray) -> float:
    if ref.shape != enh.shape:
        enh = cv2.resize(enh, (ref.shape[1], ref.shape[0]))
    mse = np.mean((ref.astype(np.float64) - enh.astype(np.float64)) ** 2)
    return 100.0 if mse < 1e-10 else round(20 * math.log10(255 / math.sqrt(mse)), 2)


def compute_ssim(ref: np.ndarray, enh: np.ndarray) -> float:
    if ref.shape != enh.shape:
        enh = cv2.resize(enh, (ref.shape[1], ref.shape[0]))
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    o, e = ref.astype(np.float64), enh.astype(np.float64)
    mu1 = cv2.GaussianBlur(o, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(e, (11, 11), 1.5)
    s1  = cv2.GaussianBlur(o ** 2, (11, 11), 1.5) - mu1 ** 2
    s2  = cv2.GaussianBlur(e ** 2, (11, 11), 1.5) - mu2 ** 2
    s12 = cv2.GaussianBlur(o * e,  (11, 11), 1.5) - mu1 * mu2
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return round(float((num / (den + 1e-12)).mean()), 4)


def compute_sam(ref: np.ndarray, enh: np.ndarray) -> float:
    if ref.shape != enh.shape:
        enh = cv2.resize(enh, (ref.shape[1], ref.shape[0]))
    o, e = ref.astype(np.float32), enh.astype(np.float32)
    dot  = np.sum(o * e, axis=2)
    norm = np.linalg.norm(o, axis=2) * np.linalg.norm(e, axis=2)
    norm[norm < 1e-8] = 1e-8
    return round(float(np.mean(np.arccos(np.clip(dot / norm, -1, 1)))) * 180 / math.pi, 4)


def laplacian_sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def spectral_consistency(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    scores = []
    for c in range(a.shape[2]):
        h1 = cv2.calcHist([a[:, :, c]], [0], None, [64], [0, 256]).flatten()
        h2 = cv2.calcHist([b[:, :, c]], [0], None, [64], [0, 256]).flatten()
        h1 /= h1.sum() + 1e-8
        h2 /= h2.sum() + 1e-8
        scores.append(np.clip(float(np.sum(np.sqrt(h1 * h2 + 1e-12))), 0, 1))
    return round(float(np.mean(scores)) * 100, 1)


# ══════════════════════════════════════════════════════════════════════════════
# GEOSPATIAL LAND-COVER ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
class GeospatialAnalyzer:
    """
    Pixel-level land-cover segmentation from RGB satellite imagery.

    Indices used
    ─────────────
    VARI  (Visible Atmospherically Resistant Index) for vegetation
    NDWI  proxy via blue-dominance heuristic for water bodies
    Canny edge detection + contour area filtering for built-up structures
    """

    @staticmethod
    def analyze(rgb: np.ndarray) -> dict:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        r, g, b = [rgb[:, :, i].astype(np.float32) for i in range(3)]

        # ── VARI ────────────────────────────────────────────────────────────
        denom = g + r - b
        denom[np.abs(denom) < 1e-5] = 1e-5
        vari = (g - r) / denom

        veg_mask = cv2.morphologyEx(
            np.where(vari > 0.06, 255, 0).astype(np.uint8),
            cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
        )
        healthy_mask  = np.where((veg_mask > 0) & (vari > 0.15), 255, 0).astype(np.uint8)
        stressed_mask = np.where((veg_mask > 0) & (vari <= 0.15), 255, 0).astype(np.uint8)

        # ── Water (NDWI proxy) ──────────────────────────────────────────────
        water_mask = np.where(
            (b > r + 8) & (g > r + 3) & (r < 110) & (b < 150), 255, 0
        ).astype(np.uint8)
        water_mask = cv2.morphologyEx(
            water_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
        )

        # ── Built-up / Urban (Canny) ────────────────────────────────────────
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 40, 120)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bldgs   = [c for c in cnts if 8 < cv2.contourArea(c) < gray.size * 0.05]

        build_mask = np.zeros_like(gray)
        edge_mask  = np.zeros_like(gray)
        cv2.drawContours(build_mask, bldgs, -1, 255, -1)
        cv2.drawContours(edge_mask,  bldgs, -1, 255,  2)

        total = gray.size
        return {
            "vegetation_percent":    round(np.sum(veg_mask     > 0) / total * 100, 1),
            "healthy_crop_percent":  round(np.sum(healthy_mask  > 0) / total * 100, 1),
            "stressed_crop_percent": round(np.sum(stressed_mask > 0) / total * 100, 1),
            "water_percent":         round(np.sum(water_mask    > 0) / total * 100, 1),
            "buildup_percent":       round(np.sum(build_mask    > 0) / total * 100, 1),
            "detected_structures":   len(bldgs),
            # masks (internal, for overlay rendering)
            "healthy_mask":  healthy_mask,
            "stressed_mask": stressed_mask,
            "water_mask":    water_mask,
            "build_mask":    build_mask,
            "edge_mask":     edge_mask,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SUPER-RESOLUTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class SuperResolutionEngine:
    """
    Primary  : A2N  — pretrained model (HuggingFace / super-image library)
               Attention-based Attention Network, trained on DIV2K ×4
    Fallback : EDSR-Lite  — LANCZOS4 upscale + detail-enhance + CLAHE
    """

    def __init__(self):
        self.scale_factor = SCALE_FACTOR
        self._si_model    = None
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        # Priority 1: Our satellite-trained EDSR weights
        sat_path = MODELS_DIR / "edsr_satellite.pth" if MODELS_DIR else None
        if sat_path and sat_path.exists():
            logger.info(f"Loading satellite-trained EDSR from {sat_path}…")
            self._build_edsr_lite()   # build architecture first
            try:
                import torch
                self._edsr_net.load_state_dict(
                    torch.load(str(sat_path), map_location=self._edsr_dev)
                )
                self._edsr_net.eval()
                self.model_name = "EDSR-Satellite (SIH 2024 · Sentinel-2 ×4)"
                self.model_type = "EDSR-SAT"
                self._si_model  = None
                logger.info("✓ Satellite-trained EDSR weights loaded")
                return
            except Exception as exc:
                logger.warning(f"Failed to load satellite weights ({exc}) — falling back")

        # Priority 2: A2N from HuggingFace (pretrained on DIV2K)
        try:
            from super_image import A2nModel
            logger.info("Downloading / loading A2N from HuggingFace…")
            self._si_model  = A2nModel.from_pretrained("eugenesiow/a2n", scale=4)
            self._si_model.eval()
            self.model_name = "A2N (Pretrained · DIV2K ×4)"
            self.model_type = "A2N"
            logger.info("✓ A2N model ready")
        except Exception as exc:
            logger.warning(f"super-image unavailable ({exc}) — EDSR-Lite fallback active")
            self.model_name = "EDSR-Lite (Algorithmic ×4)"
            self.model_type = "EDSR-LITE"
            self._build_edsr_lite()

    def _build_edsr_lite(self):
        """Lightweight EDSR fallback (no pretrained weights needed)."""
        import torch
        import torch.nn as nn

        class ResBlock(nn.Module):
            def __init__(self, f=32):
                super().__init__()
                self.b = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(True), nn.Conv2d(f, f, 3, padding=1))
            def forward(self, x): return x + self.b(x) * 0.1

        class EDSR(nn.Module):
            def __init__(self, f=32, nb=8, scale=4):
                super().__init__()
                self.head = nn.Conv2d(3, f, 3, padding=1)
                self.body = nn.Sequential(*[ResBlock(f) for _ in range(nb)], nn.Conv2d(f, f, 3, padding=1))
                self.tail = nn.Sequential(nn.Conv2d(f, f * (scale ** 2), 3, padding=1), nn.PixelShuffle(scale), nn.Conv2d(f, 3, 3, padding=1))
            def forward(self, x):
                h = self.head(x)
                return self.tail(self.body(h) + h)

        import torch.backends.mps as _mps
        if _mps.is_available():
            dev = torch.device("mps")
        elif __import__("torch").cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        self._edsr_dev  = dev
        self._edsr_net  = EDSR().to(dev).eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    def enhance(self, img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._si_model is not None:
            return self._a2n_enhance(img_rgb)
        return self._edsr_enhance(img_rgb)

    def _a2n_enhance(self, img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from super_image import ImageLoader
        from PIL import Image
        import torch

        pil    = Image.fromarray(img_rgb)
        inputs = ImageLoader.load_image(pil)

        with torch.no_grad():
            preds = self._si_model(inputs)

        sr = (
            preds.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .float()
            .numpy()
        )
        sr = np.clip(sr * 255, 0, 255).astype(np.uint8)

        # Gentle CLAHE to restore any over-compressed spectral detail
        lab            = cv2.cvtColor(sr, cv2.COLOR_RGB2LAB)
        clahe          = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        lab[:, :, 0]   = clahe.apply(lab[:, :, 0])
        sr             = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return sr, self._confidence_map(img_rgb, sr)

    def _edsr_enhance(self, img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch
        h, w = img_rgb.shape[:2]
        # LANCZOS4 upscale (gold standard for SR baseline)
        up    = cv2.resize(
            img_rgb,
            (w * self.scale_factor, h * self.scale_factor),
            interpolation=cv2.INTER_LANCZOS4,
        )
        # Edge-preserving detail enhance
        sharp = cv2.detailEnhance(up, sigma_s=10, sigma_r=0.15)
        # CLAHE on luminance channel
        lab          = cv2.cvtColor(sharp, cv2.COLOR_RGB2LAB)
        clahe        = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        sr           = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return sr, self._confidence_map(img_rgb, sr)

    @staticmethod
    def _confidence_map(lr: np.ndarray, sr: np.ndarray) -> np.ndarray:
        """
        Per-pixel reconstruction confidence.
        High confidence (bright) = region closely matches upscaled input.
        Low confidence (dark)    = AI-hallucinated detail — verify against ground truth.
        """
        lr_up      = cv2.resize(lr, (sr.shape[1], sr.shape[0]),
                                interpolation=cv2.INTER_CUBIC)
        diff       = cv2.cvtColor(cv2.absdiff(lr_up, sr), cv2.COLOR_RGB2GRAY)
        local_var  = cv2.GaussianBlur(diff.astype(np.float32), (15, 15), 5)
        conf       = np.clip(255 - local_var * 3, 0, 255)
        return conf.astype(np.uint8)


# ── Singleton instances ───────────────────────────────────────────────────────
sr_engine    = SuperResolutionEngine()
geo_analyzer = GeospatialAnalyzer()
logger.info(f"SR engine: {sr_engine.model_name}")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO SCENE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_demo_scene(path: Path, scene: str) -> None:
    """Generate a realistic synthetic Sentinel-2 RGB scene."""
    seed_map = {"agriculture": 42, "urban": 17, "coastal": 99}
    rng  = np.random.default_rng(seed_map.get(scene, 0))
    img  = np.zeros((256, 256, 3), dtype=np.uint8)

    if scene == "agriculture":
        # Alternating crop strips with varying health
        palettes = [
            [34, 88, 28], [52, 115, 40], [26, 72, 20],
            [68, 130, 50], [44, 98, 35], [80, 145, 62],
        ]
        for i, y in enumerate(range(0, 256, 36)):
            img[y:y + 36, :] = palettes[i % len(palettes)]
        # Field boundaries
        for y in range(0, 256, 36):
            img[y:y + 2, :] = [18, 44, 14]
        for x in range(0, 256, 32):
            img[:, x:x + 1] = [18, 44, 14]
        # Irrigation canal
        img[108:115, :] = [52, 100, 155]
        img[:, 130:137] = [52, 100, 155]
        # Bare soil patches (harvest/fallow)
        for _ in range(5):
            y0, x0 = int(rng.integers(20, 210)), int(rng.integers(20, 210))
            img[y0:y0 + 16, x0:x0 + 22] = [138, 112, 84]
        # Stressed crop zones (yellowing)
        img[40:70, 33:95]   = [120, 138, 50]
        img[150:185, 135:200] = [115, 132, 48]

    elif scene == "urban":
        img[:, :] = [182, 178, 172]
        # Road grid
        for y in range(0, 256, 42):
            img[y:y + 6, :] = [150, 148, 143]
        for x in range(0, 256, 42):
            img[:, x:x + 6] = [150, 148, 143]
        # Building footprints with varied rooftop colours
        for bx in range(8, 250, 42):
            for by in range(8, 250, 42):
                lum = int(rng.integers(105, 165))
                col = [lum, lum - 4, lum + 7]
                img[by:by + 30, bx:bx + 30] = np.clip(col, 70, 200)
                # Shadow on south/east edges
                img[by + 28:by + 31, bx:bx + 30] = np.clip([c - 30 for c in col], 40, 170)
                img[by:by + 30, bx + 28:bx + 31] = np.clip([c - 30 for c in col], 40, 170)
        # Park
        img[6:52, 195:250]  = [52, 112, 40]
        # Water feature
        img[200:235, 8:60]  = [48, 96, 155]

    elif scene == "coastal":
        # Water gradient (ocean left → right)
        for x in range(128):
            frac = x / 127
            img[:, x] = np.clip(
                [int(22 + 28 * frac), int(65 + 30 * frac), int(115 + 40 * frac)],
                0, 255,
            )
        # Sandy beach transition
        img[:, 115:135] = [198, 178, 132]
        # Inland vegetation
        img[:, 135:] = [52, 112, 40]
        # River mouth cutting into land
        img[95:130, 85:145] = [55, 105, 158]
        # Coastal settlement
        for bx in range(140, 245, 22):
            for by in range(55, 205, 22):
                lum = int(rng.integers(128, 162))
                img[by:by + 15, bx:bx + 15] = [lum, lum - 4, lum + 5]
        # Flood-inundated zone (disaster context)
        img[170:215, 135:195] = [60, 110, 158]

    # Sensor-noise simulation
    noise = rng.integers(-8, 9, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img   = cv2.GaussianBlur(img, (3, 3), 0.8)   # atmospheric blurring

    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    logger.info(f"Demo scene generated: {scene} → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(
    lr_img: np.ndarray,
    hr_ref: Optional[np.ndarray],
    meta:   dict,
    label:  str = "",
) -> dict:
    t0 = time.time()

    # 1 — Super-resolution ────────────────────────────────────────────────────
    sr_img, conf_map = sr_engine.enhance(lr_img)
    proc_time = round(time.time() - t0, 2)

    # 2 — Quality metrics ─────────────────────────────────────────────────────
    if hr_ref is not None:
        ref = cv2.resize(hr_ref, (sr_img.shape[1], sr_img.shape[0]))
        metrics = {
            "psnr":               compute_psnr(ref, sr_img),
            "ssim":               compute_ssim(ref, sr_img),
            "sam":                compute_sam(ref, sr_img),
            "benchmark_mode":     True,
            "reference_available": True,
        }
    else:
        baseline = cv2.resize(
            lr_img, (sr_img.shape[1], sr_img.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        metrics = {
            "psnr":               compute_psnr(baseline, sr_img),
            "ssim":               compute_ssim(baseline, sr_img),
            "sam":                compute_sam(baseline, sr_img),
            "benchmark_mode":     False,
            "reference_available": False,
        }

    sh_lr  = laplacian_sharpness(lr_img)
    sh_sr  = laplacian_sharpness(sr_img)
    metrics.update({
        "sharpness_gain":        round(sh_sr / max(sh_lr, 1e-5), 2),
        "spectral_consistency":  spectral_consistency(lr_img, sr_img),
    })

    # 3 — Geospatial land-cover ───────────────────────────────────────────────
    geo_lr = geo_analyzer.analyze(lr_img)
    geo_sr = geo_analyzer.analyze(sr_img)

    geo_stats = {
        "vegetation_percent":    geo_sr["vegetation_percent"],
        "healthy_crop_percent":  geo_sr["healthy_crop_percent"],
        "stressed_crop_percent": geo_sr["stressed_crop_percent"],
        "water_percent":         geo_sr["water_percent"],
        "buildup_percent":       geo_sr["buildup_percent"],
        "detected_structures":   geo_sr["detected_structures"],
        "edge_improvement":      round(
            laplacian_sharpness(geo_sr["edge_mask"]) /
            max(laplacian_sharpness(geo_lr["edge_mask"]), 1e-5),
            2,
        ),
    }

    # 4 — Overlays ────────────────────────────────────────────────────────────
    conf_heat = cv2.applyColorMap(conf_map, cv2.COLORMAP_INFERNO)

    overlay = sr_img.copy()
    overlay[geo_sr["water_mask"]    > 0] = [30,  100, 250]
    overlay[geo_sr["healthy_mask"]  > 0] = [34,  139,  34]
    overlay[geo_sr["stressed_mask"] > 0] = [200, 220,  50]
    overlay[geo_sr["build_mask"]    > 0] = [255,  80,  80]
    overlay = cv2.addWeighted(sr_img, 0.58, overlay, 0.42, 0)

    return {
        "success":          True,
        "processing_time":  proc_time,
        "model_name":       sr_engine.model_name,
        "model_type":       sr_engine.model_type,
        "scale_factor":     SCALE_FACTOR,
        "scene_label":      label,
        "source_info":      meta,
        "metrics":          metrics,
        "geospatial":       geo_stats,
        "dimensions": {
            "input":  {"width": lr_img.shape[1], "height": lr_img.shape[0]},
            "output": {"width": sr_img.shape[1], "height": sr_img.shape[0]},
        },
        "images": {
            "input":       to_b64(lr_img),
            "output":      to_b64(sr_img),
            "confidence":  to_b64(conf_heat),
            "geo_overlay": to_b64(overlay),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="SRM-API · Deep Learning Super Resolution Mapping",
    description=(
        "PS-26142 | NTRO | SIH 2024\n"
        "10 m Sentinel-2 → 2.5 m via A2N ×4\n"
        "Applications: Crop Monitoring · Urban Analysis · Disaster Assessment"
    ),
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status":  "ok",
        "model":   sr_engine.model_name,
        "type":    sr_engine.model_type,
        "scale":   SCALE_FACTOR,
        "version": "2.0.0",
    }


@app.get("/api/ping")
def ping():
    """Lightweight keep-alive for frontend warm-up poll."""
    return {"alive": True, "ts": int(time.time())}


@app.post("/api/demo/{scene}")
async def demo(scene: str):
    """Process one of three preset Sentinel-2 demo scenes."""
    valid_scenes = {"agriculture", "urban", "coastal"}
    if scene not in valid_scenes:
        raise HTTPException(400, f"scene must be one of {valid_scenes}")

    demo_path = DEMO_DIR / f"s2_{scene}.png"
    if not demo_path.exists():
        generate_demo_scene(demo_path, scene)

    hr_ref, meta = load_image_any(str(demo_path))
    lr_img = cv2.resize(
        hr_ref,
        (max(8, hr_ref.shape[1] // SCALE_FACTOR),
         max(8, hr_ref.shape[0] // SCALE_FACTOR)),
        interpolation=cv2.INTER_AREA,
    )
    meta["mode"] = "demo"

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _EXECUTOR, run_pipeline, lr_img, hr_ref, meta, scene.title()
    )
    result["session_id"] = f"demo_{scene}_{int(time.time())}"
    return JSONResponse(result)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Upload any satellite image (GeoTIFF / PNG / JPG / TIFF)."""
    allowed_ext = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed_ext))}",
        )

    sid      = str(uuid.uuid4())[:8]
    up_path  = INPUT_DIR / f"{sid}{ext}"
    try:
        with open(up_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        size_kb = up_path.stat().st_size / 1024
        logger.info(f"Upload: {file.filename} ({size_kb:.1f} KB)")

        lr_img, meta = load_image_any(str(up_path))
        if min(lr_img.shape[:2]) < 16:
            raise HTTPException(400, "Image too small (minimum 16×16 px)")

        meta.update({"filename": file.filename, "session_id": sid, "mode": "upload"})

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _EXECUTOR, run_pipeline, lr_img, None, meta, ""
        )
        result["session_id"] = sid
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload pipeline error")
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)
    finally:
        if up_path.exists():
            up_path.unlink(missing_ok=True)

frontend_dir = BASE_DIR / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    logger.warning("Frontend dist folder not found!")

# ── Serve React Frontend (same-origin, no CORS issues) ───────────────────────
_DIST = Path(__file__).parent / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str = ""):
        # Don't intercept /api/* routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        index = _DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(status_code=404, detail="Frontend not built")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

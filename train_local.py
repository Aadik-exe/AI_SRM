"""
train_local.py — Train EDSR-Satellite on your Mac (Apple MPS GPU)
Problem Statement 26142 | NTRO | SIH 2024

Run:  python3 train_local.py
Time: ~1.5–2.5 hrs on M1/M2 Mac
Out:  models/edsr_satellite.pth  (auto-loaded by app.py on next run)
"""

import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ Using Apple MPS (Mac GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✓ Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("⚠ Using CPU (slower — ~5-6 hrs)")

# ── Config ────────────────────────────────────────────────────────────────────
SCALE      = 4        # 10m → 2.5m
PATCH_SIZE = 64       # LR patch (HR = 64×4 = 256px)
BATCH      = 12       # Safe for MPS memory
EPOCHS     = 100      # ~2 hrs on MPS; increase to 150 for better quality
LR_INIT    = 1e-4
N_SCENES   = 500      # Training scenes (more = better, slower data gen)
N_VAL      = 60       # Validation scenes

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
OUT_PATH   = MODELS_DIR / "edsr_satellite.pth"
CACHE_DIR  = Path(__file__).parent / "data" / "train_cache"

print(f"\nConfig: {EPOCHS} epochs · batch {BATCH} · scale {SCALE}×")
print(f"Output: {OUT_PATH}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC SENTINEL-2 SCENE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def make_scene(size: int = 512, kind: str = None, seed: int = None) -> np.ndarray:
    """Generate a realistic synthetic Sentinel-2 RGB scene (uint8)."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    kinds = ["agri", "urban", "forest", "coastal", "arid", "mixed"]
    k = kind or random.choice(kinds)
    img = np.zeros((size, size, 3), dtype=np.float32)

    if k == "agri":
        pals = [
            [34, 90, 28], [55, 120, 42], [24, 72, 18],
            [70, 135, 50], [44, 100, 34], [85, 148, 62],
            [120, 130, 45], [140, 145, 50],
        ]
        sh = random.randint(28, 52)
        for i, y in enumerate(range(0, size, sh)):
            img[y : y + sh, :] = pals[i % len(pals)]
        for y in range(0, size, sh):
            img[y : y + 2, :] = [15, 40, 12]
        sw = random.randint(24, 44)
        for x in range(0, size, sw):
            img[:, x : x + 1] = [15, 40, 12]
        cy = random.randint(size // 4, 3 * size // 4)
        img[cy : cy + 8, :] = [48, 98, 158]
        cx = random.randint(size // 4, 3 * size // 4)
        img[:, cx : cx + 8] = [48, 98, 158]
        for _ in range(random.randint(3, 8)):
            y0, x0 = random.randint(0, size - 32), random.randint(0, size - 44)
            img[y0 : y0 + random.randint(15, 30), x0 : x0 + random.randint(25, 42)] = [
                random.uniform(128, 158),
                random.uniform(105, 128),
                random.uniform(75, 98),
            ]

    elif k == "urban":
        base = random.uniform(172, 196)
        img[:, :] = [base, base - 3, base - 8]
        rs = random.randint(34, 58)
        rw = random.randint(4, 9)
        for y in range(0, size, rs):
            img[y : y + rw, :] = [148, 146, 140]
        for x in range(0, size, rs):
            img[:, x : x + rw] = [148, 146, 140]
        for bx in range(rw + 2, size, rs):
            for by in range(rw + 2, size, rs):
                bw = rs - rw - 4
                if bw < 4:
                    continue
                lum = random.uniform(98, 172)
                img[by : by + bw, bx : bx + bw] = [lum, lum - 3, lum + 6]
                # Shadow
                img[by + bw - 3 : by + bw, bx : bx + bw] = [lum - 28, lum - 31, lum - 22]
        py, px = random.randint(8, size - 62), random.randint(8, size - 72)
        img[py : py + random.randint(38, 80), px : px + random.randint(52, 105)] = [50, 110, 38]
        wy, wx = random.randint(8, size - 42), random.randint(8, size - 62)
        img[wy : wy + random.randint(24, 46), wx : wx + random.randint(42, 82)] = [48, 96, 155]

    elif k == "forest":
        img[:, :] = [22, 68, 18]
        for _ in range(220):
            cy2, cx2 = random.randint(0, size), random.randint(0, size)
            r = random.randint(4, 28)
            cv2.circle(
                img, (cx2, cy2), r,
                [random.uniform(14, 58), random.uniform(52, 134), random.uniform(11, 42)],
                -1,
            )
        for _ in range(random.randint(1, 5)):
            cy2, cx2 = random.randint(50, size - 50), random.randint(50, size - 50)
            rr = random.randint(14, 44)
            cv2.ellipse(
                img, (cx2, cy2), (rr, rr // 2), random.randint(0, 180), 0, 360,
                [random.uniform(98, 152), random.uniform(118, 168), random.uniform(78, 115)],
                -1,
            )
        if random.random() > 0.45:
            pts = np.array([
                [random.randint(0, size // 4), 0],
                [random.randint(size // 4, size // 2), size // 3],
                [random.randint(size // 3, 2 * size // 3), 2 * size // 3],
                [random.randint(3 * size // 4, size - 1), size - 1],
            ], np.int32)
            cv2.polylines(img.astype(np.uint8), [pts], False,
                          [48, 96, 155], random.randint(6, 15))

    elif k == "coastal":
        for x in range(size // 2):
            f = x / (size // 2)
            img[:, x] = [20 + 28 * f, 62 + 32 * f, 112 + 42 * f]
        img[:, size // 2 - 22 : size // 2 + 16] = [196, 175, 128]
        img[:, size // 2 + 16 :] = [48, 108, 38]
        ry = random.randint(size // 4, 3 * size // 4)
        img[ry : ry + 26, size // 3 : size // 2 + 22] = [52, 108, 158]
        # Flood zone
        img[3 * size // 4 : 3 * size // 4 + 40, size // 2 + 16 : size // 2 + 70] = [60, 110, 158]

    elif k == "arid":
        img[:, :] = [185, 158, 105]
        for _ in range(18):
            cy2, cx2 = random.randint(0, size), random.randint(0, size)
            cv2.ellipse(
                img, (cx2, cy2),
                (random.randint(10, 52), random.randint(5, 28)),
                random.randint(0, 180), 0, 360,
                [random.uniform(162, 202), random.uniform(138, 172), random.uniform(88, 122)],
                -1,
            )
        if random.random() > 0.4:
            img[size // 2 - 4 : size // 2 + 4, :] = [48, 96, 155]

    else:  # mixed
        h2 = size // 2
        img[:h2, :] = [48, 108, 38]
        img[h2:, :] = [178, 172, 165]
        for y in range(0, h2, 32):
            for x in range(0, size, 28):
                lum = random.uniform(34, 98)
                img[y : y + 30, x : x + 26] = [lum * 0.44, lum, lum * 0.37]
        for bx in range(8, size - 22, 38):
            for by in range(h2 + 8, size - 22, 38):
                lum = random.uniform(118, 168)
                img[by : by + 26, bx : bx + 26] = [lum, lum - 4, lum + 5]

    img = np.clip(img, 0, 255).astype(np.uint8)
    noise = np.random.randint(-10, 11, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0.75)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class SatelliteDS(Dataset):
    def __init__(self, files, patch=64, scale=4, n_per=10, aug=True):
        self.files = files
        self.patch = patch
        self.scale = scale
        self.n     = n_per
        self.aug   = aug

    def __len__(self):
        return len(self.files) * self.n

    def __getitem__(self, idx):
        img = cv2.cvtColor(
            cv2.imread(str(self.files[idx // self.n])), cv2.COLOR_BGR2RGB
        )
        h, w = img.shape[:2]
        hp   = self.patch * self.scale
        y0   = random.randint(0, h - hp)
        x0   = random.randint(0, w - hp)
        hr   = img[y0 : y0 + hp, x0 : x0 + hp]
        lr   = cv2.resize(hr, (self.patch, self.patch), interpolation=cv2.INTER_AREA)
        if self.aug:
            if random.random() > 0.5:
                lr = np.fliplr(lr).copy();  hr = np.fliplr(hr).copy()
            if random.random() > 0.5:
                lr = np.flipud(lr).copy();  hr = np.flipud(hr).copy()
            k = random.randint(0, 3)
            lr = np.rot90(lr, k).copy();    hr = np.rot90(hr, k).copy()

        t = lambda x: torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1)
        return t(lr), t(hr)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL — EDSR-Satellite
# ══════════════════════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, f=64, rs=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f, 3, padding=1),
        )
        self.rs = rs

    def forward(self, x):
        return x + self.body(x) * self.rs


class EDSR_Satellite(nn.Module):
    def __init__(self, f=64, nb=20, scale=4, nc=3):
        super().__init__()
        self.head = nn.Conv2d(nc, f, 3, padding=1)
        body = [ResBlock(f) for _ in range(nb)] + [nn.Conv2d(f, f, 3, padding=1)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(
            nn.Conv2d(f, f * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(f, nc, 3, padding=1),
        )

    def forward(self, x):
        h = self.head(x)
        return self.tail(self.body(h) + h)


# ══════════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════════
class CharbonnierLoss(nn.Module):
    """Charbonnier loss — smoother than L1, better for satellite textures."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, sr, hr):
        return torch.mean(torch.sqrt((sr - hr) ** 2 + self.eps ** 2))


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
def psnr(sr, hr):
    mse = ((sr - hr) ** 2).mean().item()
    return 100.0 if mse < 1e-10 else 10 * math.log10(1.0 / mse)


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE / CACHE TRAINING DATA
# ══════════════════════════════════════════════════════════════════════════════
CACHE_DIR.mkdir(parents=True, exist_ok=True)
kinds = ["agri", "urban", "forest", "coastal", "arid", "mixed"]
total = N_SCENES + N_VAL

existing = list(CACHE_DIR.glob("*.png"))
if len(existing) < total:
    print(f"Generating {total} synthetic Sentinel-2 scenes (512×512)...")
    for i in range(total):
        path = CACHE_DIR / f"scene_{i:04d}_{kinds[i % len(kinds)]}.png"
        if not path.exists():
            img = make_scene(512, kinds[i % len(kinds)], seed=i)
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{total} scenes generated")
    print("✓ Dataset ready\n")
else:
    print(f"✓ Using cached {len(existing)} scenes\n")

all_files = sorted(CACHE_DIR.glob("*.png"))
tr_files  = all_files[:N_SCENES]
va_files  = all_files[N_SCENES:]

tr_dl = DataLoader(
    SatelliteDS(tr_files, n_per=10, aug=True),
    batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False
)
va_dl = DataLoader(
    SatelliteDS(va_files, n_per=4, aug=False),
    batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False
)
print(f"Train: {len(tr_dl.dataset):,} patches | Val: {len(va_dl.dataset):,} patches\n")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
model     = EDSR_Satellite().to(DEVICE)
criterion = CharbonnierLoss()
optimizer = optim.Adam(model.parameters(), lr=LR_INIT, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model:  EDSR-Satellite  ({total_params/1e6:.2f}M params)")
print(f"Loss:   Charbonnier")
print(f"Optim:  Adam  LR={LR_INIT}  cosine annealing → 1e-6")
print("=" * 62)

best_psnr = 0.0
t_start   = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    t_ep = time.time()

    for lr_b, hr_b in tr_dl:
        lr_b, hr_b = lr_b.to(DEVICE), hr_b.to(DEVICE)
        optimizer.zero_grad()
        sr_b = model(lr_b)
        loss = criterion(sr_b.clamp(0, 1), hr_b)
        loss.backward()
        # Gradient clipping — prevents MPS NaN issues
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    train_loss /= len(tr_dl)

    # Validate every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for lr_b, hr_b in va_dl:
                sr_b = model(lr_b.to(DEVICE)).clamp(0, 1)
                val_psnr += psnr(sr_b, hr_b.to(DEVICE))
        val_psnr /= len(va_dl)

        elapsed   = time.time() - t_ep
        total_ela = time.time() - t_start
        eta       = (total_ela / epoch) * (EPOCHS - epoch)
        eta_str   = f"{eta/60:.0f}min" if eta < 3600 else f"{eta/3600:.1f}hr"

        mark = ""
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), OUT_PATH)
            mark = "  ← BEST ✓"

        print(
            f"Ep {epoch:3d}/{EPOCHS} | "
            f"Loss {train_loss:.5f} | "
            f"PSNR {val_psnr:.2f} dB | "
            f"LR {scheduler.get_last_lr()[0]:.1e} | "
            f"{elapsed:.0f}s | ETA {eta_str}"
            f"{mark}"
        )
        model.train()

print("=" * 62)
print(f"\n✅ Training complete!")
print(f"   Best PSNR : {best_psnr:.2f} dB")
print(f"   Weights   : {OUT_PATH}")
print(f"   Total time: {(time.time()-t_start)/60:.1f} min")
print()
print("The backend will auto-load these weights on next startup.")
print("Restart your local server:  kill -9 $(lsof -ti:8000) && python3 app.py")

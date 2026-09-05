"""
train_ci.py — Cloud-optimized training script for GitHub Actions
Problem Statement 26142 | NTRO | SIH 2024
"""
import os, time, math, random
from pathlib import Path
import cv2, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cpu")

# Cloud CI Config (Balanced for ~2 hour CPU execution)
SCALE      = 4
PATCH_SIZE = 64
BATCH      = 8
EPOCHS     = 45
LR_INIT    = 2e-4
N_SCENES   = 200
N_VAL      = 20

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
OUT_PATH   = MODELS_DIR / "edsr_satellite.pth"
CACHE_DIR  = Path(__file__).parent / "data" / "train_cache"

print(f"Cloud CI Config: {EPOCHS} epochs · batch {BATCH} · {N_SCENES} scenes")

# ── Scene Generator ──
def make_scene(size=512, seed=0):
    np.random.seed(seed); random.seed(seed)
    img = np.zeros((size, size, 3), dtype=np.float32)
    base = random.uniform(50, 150)
    img[:, :] = [base*.4, base, base*.6]
    for _ in range(50):
        cv2.circle(img, (random.randint(0,size), random.randint(0,size)), random.randint(10,40),
                   [random.uniform(20,80), random.uniform(80,200), random.uniform(20,80)], -1)
    if random.random() > 0.5:
        cv2.rectangle(img, (100,100), (200, 400), [150,150,140], -1)
    noise = np.random.randint(-15, 16, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0.8)

# ── Dataset ──
class SatelliteDS(Dataset):
    def __init__(self, files, n_per=10, aug=True):
        self.files = files; self.n = n_per; self.aug = aug
    def __len__(self): return len(self.files) * self.n
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(str(self.files[idx // self.n])), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]; hp = PATCH_SIZE * SCALE
        y0, x0 = random.randint(0, h - hp), random.randint(0, w - hp)
        hr = img[y0:y0+hp, x0:x0+hp]
        lr = cv2.resize(hr, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
        if self.aug:
            if random.random() > 0.5: lr = np.fliplr(lr).copy(); hr = np.fliplr(hr).copy()
            if random.random() > 0.5: lr = np.flipud(lr).copy(); hr = np.flipud(hr).copy()
            k = random.randint(0, 3)
            lr = np.rot90(lr, k).copy(); hr = np.rot90(hr, k).copy()
        t = lambda x: torch.from_numpy(x.astype(np.float32)/255.0).permute(2, 0, 1)
        return t(lr), t(hr)

# ── Model (Ultra-Lite EDSR for CPU) ──
class ResBlock(nn.Module):
    def __init__(self, f=32):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(True), nn.Conv2d(f, f, 3, padding=1))
    def forward(self, x): return x + self.b(x) * 0.1

class EDSR_Lite(nn.Module):
    def __init__(self, f=32, nb=8):
        super().__init__()
        self.head = nn.Conv2d(3, f, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(f) for _ in range(nb)], nn.Conv2d(f, f, 3, padding=1))
        self.tail = nn.Sequential(nn.Conv2d(f, f * 16, 3, padding=1), nn.PixelShuffle(4), nn.Conv2d(f, 3, 3, padding=1))
    def forward(self, x):
        h = self.head(x)
        return self.tail(self.body(h) + h)

# ── Run ──
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print("Generating dataset...")
for i in range(N_SCENES + N_VAL):
    p = CACHE_DIR / f"scene_{i}.png"
    if not p.exists(): cv2.imwrite(str(p), cv2.cvtColor(make_scene(256, i), cv2.COLOR_RGB2BGR))

files = sorted(CACHE_DIR.glob("*.png"))
tr_dl = DataLoader(SatelliteDS(files[:N_SCENES], 8, True), batch_size=BATCH, shuffle=True)
va_dl = DataLoader(SatelliteDS(files[N_SCENES:], 2, False), batch_size=BATCH)

model = EDSR_Lite().to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR_INIT)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
crit = nn.L1Loss()

print(f"Starting training on {DEVICE}...")
best_psnr = 0.0
for ep in range(1, EPOCHS + 1):
    model.train(); tl = 0.0
    for lr, hr in tr_dl:
        opt.zero_grad()
        sr = model(lr)
        loss = crit(sr, hr)
        loss.backward(); opt.step(); tl += loss.item()
    sched.step(); tl /= len(tr_dl)
    
    if ep % 5 == 0 or ep == EPOCHS:
        model.eval(); vp = 0.0; n = 0
        with torch.no_grad():
            for lr, hr in va_dl:
                sr = model(lr).clamp(0, 1)
                mse = ((sr - hr) ** 2).mean().item()
                vp += 100. if mse < 1e-10 else 10 * math.log10(1./mse)
                n += 1
        vp /= n
        mark = " *BEST" if vp > best_psnr else ""
        if vp > best_psnr:
            best_psnr = vp
            torch.save(model.state_dict(), OUT_PATH)
        print(f"Epoch {ep}/{EPOCHS} | Loss: {tl:.4f} | PSNR: {vp:.2f}dB{mark}")

print(f"Training Complete! Saved weights to {OUT_PATH}")

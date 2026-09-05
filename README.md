<div align="center">
  <h1>🛰️ AI_SRM: Deep Learning Super Resolution Mapping</h1>
  <p><b>Problem Statement 26142 | NTRO | Smart India Hackathon 2024</b></p>
  
  [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Aadik-exe/AI_SRM)
</div>

<br>

## 📖 Overview
Medium-resolution satellite imagery (10m to 30m) offers broad coverage but lacks the spatial detail required for fine-scale analysis like identifying small buildings, narrow roads, or localized disaster damage. 

**AI_SRM** solves this by applying advanced deep learning techniques (Enhanced Deep Super-Resolution - EDSR) to upscale **10m Sentinel-2 imagery to 2.5m high-resolution imagery (4× scale)**. 

Beyond upscaling, the system includes a **Geospatial Land-Cover Analyzer** that processes the upscaled imagery to extract actionable intelligence (vegetation health, water bodies, and urban structures) for intelligence analysts and agricultural monitoring.

---

## ✨ Key Features
- **Satellite-Specific AI:** Model trained specifically on remote sensing data (not generic natural photos) using a Charbonnier Loss function to preserve sharp edges and accurate textures.
- **Geospatial Intelligence Engine:** Automatically calculates VARI (vegetation health), detects water bodies (NDWI proxy), and extracts building footprints (Canny edge detection).
- **Automated MLOps Pipeline:** Model training runs in the cloud via GitHub Actions. Push the code, and the cloud handles the heavy compute and auto-deploys the updated weights.
- **Fail-safe Fallback:** If the primary AI model fails due to memory limits, the system seamlessly degrades to an algorithmic `LANCZOS4` + `CLAHE` enhancement pipeline.
- **Production-Ready UI:** Highly interactive, military-intelligence-styled dashboard built with React and Vite.

---

## 🏗️ Architecture Stack
* **Frontend:** React, TypeScript, Vite, Tailwind CSS, Lucide Icons.
* **Backend:** FastAPI (Python), Uvicorn, Asynchronous ThreadPool Executor.
* **AI & Vision:** PyTorch, OpenCV, Rasterio, NumPy.
* **Deployment:** Vercel (Frontend Global CDN), Render (Backend API), GitHub Actions (MLOps CI/CD).

---

## 🚀 Local Setup & Development

### 1. Backend API
```bash
# Clone the repository
git clone https://github.com/Aadik-exe/AI_SRM.git
cd AI_SRM

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
python app.py
# Server runs on http://localhost:8000
```

### 2. Frontend Dashboard
```bash
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
# Dashboard runs on http://localhost:5173
```

---

## 🧠 Model Training (MLOps)
You do not need a high-end GPU on your local machine to train the model. We have automated this using GitHub Actions.

1. Any changes pushed to `train_ci.py` automatically trigger the GitHub Actions cloud runner.
2. The cloud server creates a synthetic satellite dataset and trains the EDSR model for 45 epochs.
3. Once training completes (~2 hours), the GitHub Actions Bot commits the new `edsr_satellite.pth` weights directly to this repository.
4. Render detects the new commit and automatically redeploys the backend API with the smarter AI model.

---

## 📊 Evaluation Metrics
Our pipeline actively calculates the following metrics for NTRO validation:
- **PSNR (Peak Signal-to-Noise Ratio):** Measures pixel-level fidelity against ground-truth HR imagery.
- **SSIM (Structural Similarity Index):** Ensures structural preservation (roads, edges) remains intact.
- **SAM (Spectral Angle Mapper):** Ensures color/spectral profiles are not distorted by the AI hallucination.

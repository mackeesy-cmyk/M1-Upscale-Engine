# M1-Upscale-Engine Installation Guide

## ⚠️ Python Version Compatibility Notice

The Real-ESRGAN model dependencies (`basicsr` and `realesrgan`) currently have **compatibility issues with Python 3.13**.

### Current Status

✅ **Working**: The application runs with a **high-quality bicubic upscaling fallback** that:
- Uses OpenCV's INTER_CUBIC interpolation (better than simple bicubic)
- Applies smart sharpening for improved perceived quality
- Provides 4x resolution increase
- Works on M1 with MPS acceleration for supported operations

### For Real-ESRGAN AI Upscaling

To use the full Real-ESRGAN AI model, you have two options:

#### Option 1: Use Python 3.10 or 3.11 (Recommended)

```bash
# Install Python 3.11 via Homebrew
brew install python@3.11

# Create virtual environment with Python 3.11
/opt/homebrew/bin/python3.11 -m venv backend/venv

# Activate and install
cd backend
source venv/bin/activate
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
pip install -r requirements.txt
```

#### Option 2: Use Current Setup (Python 3.13 + Fallback)

The current installation works out of the box with high-quality upscaling, just without the AI enhancement.

## Quick Start (Current Setup)

```bash
# Backend (already installed)
cd backend
source venv/bin/activate
python main.py

# Frontend (new terminal)
cd frontend
npm run dev
```

Open `http://localhost:3000` and start upscaling!

## What's Been Installed

✅ FastAPI backend with async processing  
✅ PyTorch with MPS support  
✅ OpenCV for image processing  
✅ FFmpeg with h264_videotoolbox encoding  
✅ Next.js 14 frontend with dark UI  
✅ All core dependencies  

The only missing piece is the optional Real-ESRGAN AI model, which requires Python 3.10/3.11.

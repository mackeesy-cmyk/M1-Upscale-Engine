# M1-Upscale-Engine Installation Status

## Current Status: Fully Working

All components are installed and operational, including Real-ESRGAN AI upscaling.

### Environment

- **Python**: 3.11.14 (via Homebrew)
- **PyTorch**: 2.3.1 with MPS (Metal) acceleration
- **Real-ESRGAN**: Loaded with RealESRGAN_x4plus weights (67 MB)
- **FFmpeg**: 8.0.1 with VideoToolbox hardware encoding

### What's Installed

- FastAPI backend with async processing
- PyTorch with MPS support
- Real-ESRGAN AI model (basicsr + realesrgan)
- OpenCV for image processing
- FFmpeg with h264_videotoolbox encoding
- Next.js frontend with dark UI
- All core dependencies

## Quick Start

```bash
# Backend
cd backend
source venv/bin/activate
python main.py

# Frontend (new terminal)
cd frontend
npm run dev
```

Open `http://localhost:3000` and start upscaling!

## Note on Python Version

The virtual environment uses **Python 3.11** because `basicsr` and `realesrgan` are incompatible with Python 3.13+. Do not recreate the venv with a newer Python version.

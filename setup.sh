#!/bin/bash

echo "🚀 M1-Upscale-Engine Setup Script"
echo "=================================="
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This application is designed for macOS with Apple Silicon"
    exit 1
fi

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  Warning: This system doesn't appear to be Apple Silicon (M1/M2/M3)"
    echo "   The application will work but won't benefit from MPS acceleration"
fi

echo "📋 Checking prerequisites..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10 or later."
    exit 1
else
    echo "✓ Python 3 found: $(python3 --version)"
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18 or later."
    exit 1
else
    echo "✓ Node.js found: $(node --version)"
fi

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ FFmpeg not found."
    echo "   Install with: brew install ffmpeg"
    exit 1
else
    echo "✓ FFmpeg found: $(ffmpeg -version | head -n1)"
fi

echo ""
echo "🔧 Setting up backend..."
echo ""

cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install PyTorch with MPS support
echo "Installing PyTorch with MPS support (this may take a few minutes)..."
pip3 install --quiet --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# Install other dependencies
echo "Installing backend dependencies..."
pip3 install --quiet -r requirements.txt

echo "✓ Backend setup complete"
echo ""

cd ..

echo "🎨 Setting up frontend..."
echo ""

cd frontend

# Install npm dependencies
echo "Installing frontend dependencies..."
npm install --silent

echo "✓ Frontend setup complete"
echo ""

cd ..

echo "✅ Setup complete!"
echo ""
echo "📖 Next steps:"
echo ""
echo "1. Start the backend server:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "2. In a new terminal, start the frontend:"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "Happy upscaling! 🎬✨"

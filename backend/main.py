import os
import asyncio
import uuid
import shutil
import logging
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

from upscaler import load_model, upscale_frames
from video_utils import extract_frames, stitch_video, get_video_fps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
task_store: Dict[str, Dict[str, Any]] = {}
upscaler_model = None
device = None
processing_sem = None
TEMP_DIR = Path(__file__).parent / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global upscaler_model, device, processing_sem
    
    # Initialize semaphore
    processing_sem = asyncio.Semaphore(1)
    
    logger.info("🚀 Starting M1-Upscale-Engine...")
    
    # Load model on startup
    try:
        upscaler_model, device = await asyncio.to_thread(load_model)
        logger.info(f"✓ Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        logger.warning("Model will be loaded on first request")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down, cleaning up temporary files...")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


# Initialize FastAPI app
app = FastAPI(
    title="M1 Upscale Engine",
    description="Hardware-accelerated video upscaling for Apple Silicon",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint with MPS detection."""
    mps_available = torch.backends.mps.is_available()
    
    return {
        "status": "healthy",
        "mps_available": mps_available,
        "device": str(device) if device else "not_loaded",
        "model_loaded": upscaler_model is not None
    }


@app.get("/modes")
async def get_available_modes():
    """
    Get available upscaling modes.
    
    Returns:
        JSON with available modes and their status
    """
    ai_available = upscaler_model is not None
    
    return {
        "modes": [
            {
                "id": "fast",
                "name": "Fast Mode",
                "description": "OpenCV bicubic interpolation with sharpening",
                "available": True,
                "speed": "Fast"
            },
            {
                "id": "ai",
                "name": "AI Mode",
                "description": "Real-ESRGAN neural network upscaling",
                "available": ai_available,
                "speed": "Slower, Higher Quality"
            }
        ],
        "default": "ai" if ai_available else "fast"
    }


@app.post("/upload")
async def upload_video(file: UploadFile = File(...), mode: str = "auto"):
    """
    Upload a video file and start upscaling process.
    
    Args:
        file: Video file to upscale
        mode: Upscaling mode - 'ai', 'fast', or 'auto' (default)
    
    Returns:
        JSON with task_id for tracking progress
    """
    global upscaler_model, device
    
    # Validate file type by content_type or extension
    ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ''
    
    is_valid = (
        (file.content_type and file.content_type.startswith("video/")) or
        file_ext in ALLOWED_EXTENSIONS
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=400, 
            detail=f"File must be a video. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    
    logger.info(f"Received {file.filename} ({file.content_type}, {file_ext})")
    
    # Determine actual mode to use
    if mode == "auto":
        actual_mode = "ai" if upscaler_model is not None else "fast"
    elif mode == "ai" and upscaler_model is None:
        raise HTTPException(status_code=400, detail="AI mode not available - model not loaded")
    else:
        actual_mode = mode
    
    logger.info(f"Upload request with mode: {mode} (using: {actual_mode})")
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Create task directory
    task_dir = TEMP_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    input_path = task_dir / "input_video.mp4"
    
    try:
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Task {task_id}: Video uploaded ({len(content)} bytes)")
        
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save video")
    
    # Initialize task in store
    task_store[task_id] = {
        "status": "queued", # Initial status is queued
        "progress": 0,
        "current_frame": 0,
        "total_frames": 0,
        "input_path": str(input_path),
        "output_path": None,
        "error": None,
        "mode": None  # Will be set during processing
    }
    
    # Start background processing
    asyncio.create_task(process_video(task_id, input_path, task_dir, actual_mode))
    
    return {"task_id": task_id}


async def process_video(task_id: str, input_path: Path, task_dir: Path, mode: str = "fast"):
    """
    Background task to process video through upscaling pipeline.
    
    Pipeline:
    1. Extract frames
    2. Upscale frames with Real-ESRGAN
    3. Stitch frames back into video
    """
    global upscaler_model, device, processing_sem
    
    # Wait for slot
    async with processing_sem:
        # Update status to processing once lock acquired
        task_store[task_id]["status"] = "processing"
        
        frames_dir = task_dir / "frames"
        upscaled_dir = task_dir / "upscaled"
        output_path = task_dir / "output_upscaled.mp4"
        
        try:
            # Load model if not already loaded
            if upscaler_model is None or device is None:
                logger.info(f"Task {task_id}: Loading model...")
                # Loading model can be slow, run in thread
                upscaler_model, device = await asyncio.to_thread(load_model)
            
            # Step 1: Extract frames
            logger.info(f"Task {task_id}: Extracting frames...")
            task_store[task_id]["progress"] = 5
            # Run blocking extraction in thread
            total_frames = await asyncio.to_thread(extract_frames, input_path, frames_dir)
            task_store[task_id]["total_frames"] = total_frames
            task_store[task_id]["progress"] = 10
            
            # Step 2: Upscale frames
            logger.info(f"Task {task_id}: Upscaling {total_frames} frames...")
            # Run blocking upscaling in thread
            await asyncio.to_thread(
                upscale_frames,
                frames_dir=frames_dir,
                output_dir=upscaled_dir,
                task_id=task_id,
                task_store=task_store,
                model=upscaler_model,
                device=device,
                mode=mode
            )
            
            # Progress should be at 100% after upscaling
            task_store[task_id]["progress"] = 95
            
            # Step 3: Stitch video
            logger.info(f"Task {task_id}: Stitching video...")
            fps = get_video_fps(input_path)
            # Run blocking stitching in thread
            await asyncio.to_thread(
                stitch_video,
                frames_dir=upscaled_dir,
                output_path=output_path,
                original_video_path=input_path,
                fps=fps
            )
            
            task_store[task_id]["status"] = "complete"
            task_store[task_id]["progress"] = 100
            task_store[task_id]["output_path"] = str(output_path)
            logger.info(f"Task {task_id}: ✓ Processing complete!")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            task_store[task_id]["status"] = "error"
            task_store[task_id]["error"] = str(e)


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Get processing status for a task.
    
    Returns:
        JSON with progress, status, and frame counts
    """
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_store[task_id]
    
    return {
        "status": task["status"],
        "progress": task["progress"],
        "current_frame": task["current_frame"],
        "total_frames": task["total_frames"],
        "error": task["error"],
        "mode": task.get("mode", "Unknown")
    }


@app.get("/download/{task_id}")
async def download_video(task_id: str):
    """
    Download the upscaled video file.
    
    Returns:
        File stream of upscaled video
    """
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_store[task_id]
    
    if task["status"] != "complete":
        raise HTTPException(status_code=400, detail="Task not complete yet")
    
    output_path = Path(task["output_path"])
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"upscaled_{task_id}.mp4"
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

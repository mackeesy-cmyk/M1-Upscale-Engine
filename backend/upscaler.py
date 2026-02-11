import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import cv2
import requests

logger = logging.getLogger(__name__)

# Model configuration
WEIGHTS_DIR = Path(__file__).parent / "weights"
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_PATH = WEIGHTS_DIR / "RealESRGAN_x4plus.pth"


def download_weights() -> Path:
    """
    Download Real-ESRGAN model weights if they don't exist.
    
    Returns:
        Path to the downloaded weights file
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if MODEL_PATH.exists():
        logger.info(f"Model weights already exist at {MODEL_PATH}")
        return MODEL_PATH
    
    logger.info(f"Downloading model weights from {MODEL_URL}...")
    
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024) == 0:  # Log every MB
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")
        
        logger.info(f"Model weights downloaded successfully to {MODEL_PATH}")
        return MODEL_PATH
        
    except Exception as e:
        logger.error(f"Failed to download model weights: {e}")
        raise


def load_model() -> tuple[Any, torch.device]:
    """
    Load Real-ESRGAN model with M1 MPS optimization.
    Falls back to simple upscaling if Real-ESRGAN is not available.
    
    Returns:
        Tuple of (model, device)
    """
    # Detect best available device (prioritize MPS for M1)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("✓ MPS (Metal Performance Shaders) device detected - M1 acceleration enabled")
    else:
        device = torch.device('cpu')
        logger.warning("⚠ MPS not available - falling back to CPU (slower performance)")
    
    # Try to load Real-ESRGAN
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        
        # Download weights if needed
        weights_path = download_weights()
        
        # Define the model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        # Create upsampler with device
        upsampler = RealESRGANer(
            scale=4,
            model_path=str(weights_path),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,  # Don't use FP16 on MPS for now
            device=str(device)
        )
        
        logger.info("✓ Real-ESRGAN model loaded successfully (AI Mode available)")
        return upsampler, device
        
    except Exception as e:
        logger.warning(f"⚠ Failed to load Real-ESRGAN model: {e}")
        logger.warning("⚠ Using fallback upscaling method (OpenCV bicubic)")
        logger.info("AI Mode is not available - only Fast Mode will work")
        return None, device


class SimpleUpscaler:
    """
    Simple upscaler using OpenCV's bicubic interpolation as a fallback.
    For production use, replace with Real-ESRGAN model.
    """
    def __init__(self, scale=4):
        self.scale = scale
    
    def enhance(self, img: np.ndarray, outscale: int = 4):
        """
        Upscale image using bicubic interpolation.
        
        Args:
            img: Input image as numpy array
            outscale: Upscaling factor
            
        Returns:
            Tuple of (upscaled_image, None)
        """
        h, w = img.shape[:2]
        new_h, new_w = h * outscale, w * outscale
        
        # Use INTER_CUBIC for better quality than INTER_LINEAR
        upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply slight sharpening to improve perceived quality
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel * 0.3)
        
        # Blend original upscaled with sharpened version
        output = cv2.addWeighted(upscaled, 0.7, sharpened, 0.3, 0)
        
        return output, None


def upscale_frames(
    frames_dir: Path,
    output_dir: Path,
    task_id: str,
    task_store: Dict[str, Any],
    model: Optional[Any] = None,
    device: Optional[torch.device] = None,
    mode: str = 'fast'
) -> None:
    """
    Upscale all frames in a directory using either AI or Fast mode.
    
    Args:
        frames_dir: Directory containing input frames
        output_dir: Directory to save upscaled frames
        task_id: Unique task identifier for progress tracking
        task_store: Dictionary to update with progress information
        model: Pre-loaded model (RealESRGANer or None for fallback)
        device: PyTorch device (for logging purposes)
        mode: 'ai' for Real-ESRGAN, 'fast' for bicubic
    """
    # Determine which upscaler to use
    use_ai = mode == 'ai' and model is not None
    
    if use_ai:
        logger.info(f"Task {task_id}: Using AI Mode (Real-ESRGAN)")
    else:
        logger.info(f"Task {task_id}: Using Fast Mode (OpenCV bicubic)")
        # Create simple upscaler for fallback
        simple_upscaler = SimpleUpscaler(scale=4)
    
    # Get sorted list of frame files
    frame_files = sorted(
        [f for f in frames_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']],
        key=lambda x: int(''.join(filter(str.isdigit, x.stem)) or 0)
    )
    
    total_frames = len(frame_files)
    logger.info(f"Processing {total_frames} frames for task {task_id}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update task store
    task_store[task_id]["total_frames"] = total_frames
    task_store[task_id]["current_frame"] = 0
    task_store[task_id]["mode"] = "AI Mode" if use_ai else "Fast Mode"
    
    # Process frames
    for idx, frame_path in enumerate(frame_files):
        try:
            # Read image
            img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Failed to read frame {frame_path}, skipping")
                continue
            
            # Upscale based on mode
            try:
                if use_ai:
                    # Use Real-ESRGAN
                    output, _ = model.enhance(img, outscale=4)
                else:
                    # Use bicubic fallback
                    output, _ = simple_upscaler.enhance(img, outscale=4)
            except Exception as e:
                logger.error(f"Error upscaling frame {idx}: {e}")
                # Just copy original if upscaling fails
                output = img
            
            # Save upscaled frame
            output_path = output_dir / frame_path.name
            cv2.imwrite(str(output_path), output)
            
            # Update progress
            current_frame = idx + 1
            progress = int((current_frame / total_frames) * 85) + 10  # Map to 10-95% range
            
            task_store[task_id]["current_frame"] = current_frame
            task_store[task_id]["progress"] = progress
            
            if current_frame % 10 == 0 or current_frame == total_frames:
                logger.info(f"Task {task_id}: {current_frame}/{total_frames} frames ({progress}%)")
                
        except Exception as e:
            logger.error(f"Error processing frame {frame_path}: {e}")
            task_store[task_id]["error"] = str(e)
            task_store[task_id]["status"] = "error"
            raise
    
    logger.info(f"Task {task_id}: All frames upscaled successfully")
    task_store[task_id]["progress"] = 95  # Leave room for encoding stage

import subprocess
import logging
from pathlib import Path
from typing import Optional
import json

logger = logging.getLogger(__name__)


def get_video_fps(video_path: Path) -> float:
    """
    Extract FPS from video using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        FPS as float (defaults to 30.0 if detection fails)
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'json',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Parse frame rate (format: "30/1" or "30000/1001")
        fps_str = data['streams'][0]['r_frame_rate']
        num, denom = map(int, fps_str.split('/'))
        fps = num / denom
        
        logger.info(f"Detected FPS: {fps}")
        return fps
        
    except Exception as e:
        logger.warning(f"Failed to detect FPS, defaulting to 30: {e}")
        return 30.0


def extract_frames(video_path: Path, output_dir: Path) -> int:
    """
    Extract all frames from video using FFmpeg.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        
    Returns:
        Total number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_pattern = str(output_dir / "frame_%06d.png")
    
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-qscale:v', '1',  # High quality frames
        frame_pattern,
        '-y'  # Overwrite existing files
    ]
    
    logger.info(f"Extracting frames from {video_path}...")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Count extracted frames
        frames = list(output_dir.glob("frame_*.png"))
        total_frames = len(frames)
        
        logger.info(f"Extracted {total_frames} frames to {output_dir}")
        return total_frames
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg frame extraction failed: {e.stderr.decode()}")
        raise


def stitch_video(
    frames_dir: Path,
    output_path: Path,
    original_video_path: Path,
    fps: Optional[float] = None
) -> None:
    """
    Stitch frames back into video using FFmpeg with M1 hardware acceleration.
    
    Uses h264_videotoolbox encoder to leverage M1 Media Engine for fast encoding.
    Falls back to libx264 if hardware encoder is unavailable.
    
    Args:
        frames_dir: Directory containing upscaled frames
        output_path: Path for output video file
        original_video_path: Path to original video (for audio extraction)
        fps: Frame rate (if None, will auto-detect from original)
    """
    if fps is None:
        fps = get_video_fps(original_video_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    frame_pattern = str(frames_dir / "frame_%06d.png")
    
    # Try h264_videotoolbox first (M1 hardware acceleration)
    cmd_videotoolbox = [
        'ffmpeg',
        '-r', str(fps),
        '-i', frame_pattern,
        '-i', str(original_video_path),
        '-map', '0:v:0',  # Video from frames
        '-map', '1:a:0?',  # Audio from original (optional)
        '-c:v', 'h264_videotoolbox',
        '-b:v', '5000k',
        '-c:a', 'copy',
        '-pix_fmt', 'yuv420p',
        str(output_path),
        '-y'
    ]
    
    logger.info(f"Stitching video with h264_videotoolbox (M1 acceleration)...")
    
    try:
        subprocess.run(cmd_videotoolbox, check=True, capture_output=True)
        logger.info(f"✓ Video stitched successfully with hardware acceleration: {output_path}")
        return
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"h264_videotoolbox failed, falling back to libx264: {e.stderr.decode()}")
    
    # Fallback to software encoding
    cmd_libx264 = [
        'ffmpeg',
        '-r', str(fps),
        '-i', frame_pattern,
        '-i', str(original_video_path),
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-c:a', 'copy',
        '-pix_fmt', 'yuv420p',
        str(output_path),
        '-y'
    ]
    
    logger.info("Stitching video with libx264 (software encoding)...")
    
    try:
        subprocess.run(cmd_libx264, check=True, capture_output=True)
        logger.info(f"Video stitched successfully with software encoder: {output_path}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg stitching failed: {e.stderr.decode()}")
        raise


def get_video_info(video_path: Path) -> dict:
    """
    Get comprehensive video information using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video metadata
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration:stream=width,height,codec_name,r_frame_rate',
            '-of', 'json',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        return {
            'duration': float(data['format'].get('duration', 0)),
            'streams': data.get('streams', [])
        }
        
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {}

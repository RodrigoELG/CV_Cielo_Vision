# src/__init__.py
"""
Face Analysis Pipeline - Production-Grade Face Detection & Attribute Analysis

A modular, high-performance pipeline for real-time facial analysis:
- YOLOv8 face detection (GPU-accelerated)
- SORT/ByteTrack multi-object tracking
- DeepFace age/gender analysis (with intelligent caching)
- JSONL logging & video output
- Configurable via CLI/YAML/environment variables

Architecture:
    FrameGrabber (Thread) → Queue → Main Loop:
        Detect (YOLO) → Track (SORT) → Analyze (DeepFace) → Render & Log

Usage:
    # Simple usage
    from src import FacePipeline, load_config
    
    config = load_config()
    pipeline = FacePipeline(config)
    pipeline.run()
    
    # Advanced usage
    from src import (
        FaceDetectorTracker,
        AttributeAnalyzer,
        FrameGrabber,
        RendererLogger
    )
    
    detector = FaceDetectorTracker("weights/yolo.pt", device="cuda")
    analyzer = AttributeAnalyzer(analyze_interval_frames=15)
    # ... custom pipeline logic ...

Author: [Your Name/Team]
License: MIT
Repository: https://github.com/yourusername/face-analysis-pipeline
"""

__version__ = "1.0.0"
__author__ = "Face Analysis Pipeline Team"
__license__ = "MIT"
__all__ = [
    # Version info
    "__version__",
    
    # Configuration
    "Config",
    "load_config",
    "get_config",
    
    # Data models
    "Detection",
    "TrackedDetection",
    "AnnotatedDetection",
    "CachedAttributes",
    "LogEntry",
    "FramePacket",
    "PipelineStats",
    
    # Core components
    "FrameGrabber",
    "FaceDetectorTracker",
    "AttributeAnalyzer",
    "RendererLogger",
    
    # Main pipeline
    "FacePipeline",
    "main",
    
    # Utilities
    "draw_label",
    "draw_bbox",
    "safe_crop",
    "calculate_iou",
    "bbox_to_xyxy",
    "bbox_to_xywh",
    "FPSCounter",
    "Timer",
]



# Import configuration
from .config import (
    Config,
    load_config,
    get_config,
)

# Import data models
from .data_models import (
    Detection,
    TrackedDetection,
    AnnotatedDetection,
    CachedAttributes,
    LogEntry,
    FramePacket,
    PipelineStats,
)

# Import core components
from .frame_grabber import FrameGrabber
from .detector_tracker import FaceDetectorTracker
from .attributes import AttributeAnalyzer
from .renderer_logger import RendererLogger

# Import main pipeline
from .main import FacePipeline, main

# Import commonly used utilities
from .utils import (
    draw_label,
    draw_bbox,
    safe_crop,
    calculate_iou,
    bbox_to_xyxy,
    bbox_to_xywh,
    FPSCounter,
    Timer,
)


# ============================================================
# Package-level convenience functions
# ============================================================

def get_version() -> str:
    """
    Get package version.
    
    Returns:
        Version string (e.g., "1.0.0")
    """
    return __version__


def print_info():
    """Print package information."""
    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    FACE ANALYSIS PIPELINE                             ║
╚═══════════════════════════════════════════════════════════════════════╝

Version:     {__version__}
Author:      {__author__}
License:     {__license__}

Components:
  • YOLOv8 Face Detection (GPU-accelerated)
  • SORT/ByteTrack Tracking (persistent IDs)
  • DeepFace Analysis (age & gender with caching)
  • JSONL Logging & Video Output

Quick Start:
  from src import FacePipeline, load_config
  
  config = load_config()
  pipeline = FacePipeline(config)
  pipeline.run()

Documentation: https://github.com/yourusername/face-analysis-pipeline
    """)


def check_dependencies() -> dict:
    """
    Check if all required dependencies are installed.
    
    Returns:
        Dict with dependency name -> (installed: bool, version: str)
    """
    dependencies = {}
    
    # Core dependencies
    required = [
        "cv2",           # opencv-python
        "numpy",
        "pydantic",
        "yaml",          # pyyaml
        "torch",         # pytorch
        "ultralytics",   # yolov8
        "deepface",
        "filterpy",      # kalman filter (if using custom tracking)
    ]
    
    for dep in required:
        try:
            if dep == "cv2":
                import cv2
                dependencies["opencv-python"] = (True, cv2.__version__)
            elif dep == "yaml":
                import yaml
                dependencies["pyyaml"] = (True, getattr(yaml, "__version__", "unknown"))
            else:
                module = __import__(dep)
                version = getattr(module, "__version__", "unknown")
                dependencies[dep] = (True, version)
        except ImportError:
            dependencies[dep if dep != "cv2" else "opencv-python"] = (False, None)
    
    return dependencies


def print_dependencies():
    """Print dependency status."""
    deps = check_dependencies()
    
    print("\n╔═══════════════════════════════════════════════════════════════════════╗")
    print("║                        DEPENDENCY CHECK                              ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝\n")
    
    all_installed = True
    
    for name, (installed, version) in sorted(deps.items()):
        if installed:
            status = "✅"
            info = f"v{version}" if version != "unknown" else "(version unknown)"
        else:
            status = "❌"
            info = "NOT INSTALLED"
            all_installed = False
        
        print(f"  {status} {name:20s} {info}")
    
    print()
    
    if not all_installed:
        print("Some dependencies are missing!")
        print("   Install with: pip install -r requirements.txt")
        print("   Or with uv: uv sync\n")
    else:
        print("All dependencies installed!\n")
    
    return all_installed


def run_diagnostics():
    """
    Run comprehensive diagnostics.
    
    Checks:
    - Dependencies
    - YOLO model
    - GPU availability
    - Camera access
    """
    print("\n╔═══════════════════════════════════════════════════════════════════════╗")
    print("║                         DIAGNOSTICS                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝\n")
    
    # 1. Check dependencies
    print("[1/4] Checking dependencies...")
    deps_ok = print_dependencies()
    
    if not deps_ok:
        return False
    
    # 2. Check YOLO model
    print("[2/4] Checking YOLO model...")
    from pathlib import Path
    model_path = Path("weights/yolov8n-face-lindevs.pt")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  Model found: {model_path} ({size_mb:.1f} MB)\n")
    else:
        print(f"  Model not found: {model_path}")
        print("     Download from: [model download link]")
        print("     Place in: weights/yolov8n-face-lindevs.pt\n")
        return False
    
    # 3. Check GPU
    print("[3/4] Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"  GPU available: {gpu_name}")
            print(f"     GPU count: {gpu_count}\n")
        else:
            print("  GPU not available (will use CPU)")
            print("     Performance: ~10x slower\n")
    except Exception as e:
        print(f" Could not check GPU: {e}\n")
    
    # 4. Check camera
    print("[4/4] Checking camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  Camera accessible")
                print(f"     Resolution: {w}×{h}\n")
            else:
                print("  Camera opened but cannot read frames\n")
            cap.release()
        else:
            print("  Cannot open camera (index 0)")
            print("     Try: python -m src.utils test_camera\n")
            return False
    except Exception as e:
        print(f"  Camera check failed: {e}\n")
        return False
    
    print("All diagnostics passed!\n")
    return True


# ============================================================
# Package initialization
# ============================================================

def _check_environment():
    """Check environment on import (silent warnings)."""
    import os
    import sys
    
    # Suppress TensorFlow warnings
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TQDM_DISABLE", "1")
    
    # Check Python version
    if sys.version_info < (3, 8):
        import warnings
        warnings.warn(
            f"Python {sys.version_info.major}.{sys.version_info.minor} detected. "
            f"Python 3.8+ recommended for best compatibility.",
            RuntimeWarning
        )


# Run environment check on import
_check_environment()


# ============================================================
# For interactive use
# ============================================================

def quick_start():
    """
    Interactive quick start guide.
    
    Helps users get started with the pipeline.
    """
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                         QUICK START GUIDE                             ║
╚═══════════════════════════════════════════════════════════════════════╝

1. Check dependencies:
   >>> from src import print_dependencies
   >>> print_dependencies()

2. Run diagnostics:
   >>> from src import run_diagnostics
   >>> run_diagnostics()

3. Run pipeline with defaults:
   >>> from src import main
   >>> main()

4. Run with custom config:
   >>> from src import FacePipeline, load_config
   >>> config = load_config()
   >>> pipeline = FacePipeline(config)
   >>> pipeline.run()

5. Command-line usage:
   $ python -m src.main --source 0 --device cuda --debug

6. Get help:
   $ python -m src.main --help

Documentation: https://github.com/yourusername/face-analysis-pipeline
    """)


# ============================================================
# CLI entry point (when using python -m src)
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Simple CLI for package-level commands
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command in ["info", "--info"]:
            print_info()
        
        elif command in ["deps", "dependencies", "--check-deps"]:
            print_dependencies()
        
        elif command in ["diag", "diagnostics", "--diagnostics"]:
            run_diagnostics()
        
        elif command in ["version", "--version", "-v"]:
            print(f"Face Analysis Pipeline v{__version__}")
        
        elif command in ["help", "--help", "-h"]:
            quick_start()
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: info, deps, diag, version, help")
            sys.exit(1)
    else:
        # No command - run main pipeline
        from .main import main
        main()
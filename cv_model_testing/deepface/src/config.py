# src/config.py
"""
Configuration management for Face Analysis Pipeline.

Supports multiple config sources with priority:
1. Command-line arguments (highest priority)
2. YAML configuration file
3. Environment variables
4. Hard-coded defaults (lowest priority)

Technical Notes:
- Uses dataclass for clean defaults and type safety
- argparse handles CLI with automatic help generation
- PyYAML for human-readable config files
- Validation prevents invalid parameter combinations
"""

import argparse
import yaml
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional, Union
from pathlib import Path

@dataclass
class Config:
    """Pipeline configuration with validated defaults."""
    
    # Video Source
    source: Union[str, int] = "0"  # Camera index or video file/RTSP URL
    width: int = 1280
    height: int = 720
    fps_target: int = 30
    
    # Detection & Tracking
    device: str = "auto"  # auto|cuda|cpu
    yolo_model: str = "/home/rodrigo/Documents/CV_Cielo_Vision/cv_model_testing/deepface/src/weights/yolov8n-face-lindevs.pt"
    min_confidence: float = 0.35
    iou_threshold: float = 0.5
    tracker_max_age: int = 30      # frames before track deletion
    tracker_min_hits: int = 3       # detections before track confirmation
    
    # Attribute Analysis
    analyze_interval_frames: int = 15  # re-analyze every N frames
    min_crop_size: int = 80            # minimum face size in pixels
    crop_padding: float = 0.25         # padding ratio around face
    deepface_backend: str = "skip"     # skip|opencv|retinaface|mtcnn
    deepface_detector_enforce: bool = False
    
    # Performance Optimization
    process_every_n: int = 1           # process every Nth frame (1 = all frames)
    frame_queue_size: int = 2
    processing_threads: int = 1        # future: parallel attribute analysis
    
    # Rendering & Logging
    draw_labels: bool = True
    show_fps: bool = True
    show_display: bool = True          # False = headless mode
    jsonl_path: str = "logs/events.jsonl"
    save_video: bool = False
    output_video_path: str = "results/output.mp4"
    
    # Debug Options
    debug: bool = False
    save_debug_crops: bool = False
    debug_crops_dir: str = "debug/crops"
    verbose: bool = False              # detailed logging
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        self._normalize()
    
    def _validate(self):
        """Validate configuration parameters."""
        errors = []
        
        # Validate numeric ranges
        if not 0 < self.min_confidence <= 1.0:
            errors.append(f"min_confidence must be in (0, 1], got {self.min_confidence}")
        
        if not 0 < self.iou_threshold <= 1.0:
            errors.append(f"iou_threshold must be in (0, 1], got {self.iou_threshold}")
        
        if self.width <= 0 or self.height <= 0:
            errors.append(f"Invalid resolution: {self.width}×{self.height}")
        
        if self.min_crop_size < 20:
            errors.append(f"min_crop_size too small: {self.min_crop_size} (minimum 20px)")
        
        if not 0 <= self.crop_padding <= 1.0:
            errors.append(f"crop_padding must be in [0, 1], got {self.crop_padding}")
        
        if self.analyze_interval_frames < 1:
            errors.append(f"analyze_interval_frames must be >= 1, got {self.analyze_interval_frames}")
        
        if self.process_every_n < 1:
            errors.append(f"process_every_n must be >= 1, got {self.process_every_n}")
        
        if self.tracker_max_age < 1:
            errors.append(f"tracker_max_age must be >= 1, got {self.tracker_max_age}")
        
        # Validate device choice
        if self.device not in ["auto", "cuda", "cpu"]:
            errors.append(f"Invalid device: {self.device} (must be auto|cuda|cpu)")
        
        # Validate backend choice
        valid_backends = ["skip", "opencv", "retinaface", "mtcnn", "ssd", "dlib"]
        if self.deepface_backend not in valid_backends:
            errors.append(f"Invalid deepface_backend: {self.deepface_backend}")
        
        if errors:
            print("Configuration Validation Errors:", file=sys.stderr)
            for err in errors:
                print(f"   • {err}", file=sys.stderr)
            sys.exit(1)
    
    def _normalize(self):
        """Normalize configuration values."""
        # Convert string source to int if it's a digit
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)
        
        # Resolve device if auto
        if self.device == "auto":
            self.device = self._detect_device()
        
        # Resolve relative paths
        self.yolo_model = str(Path(self.yolo_model).resolve())
        self.jsonl_path = str(Path(self.jsonl_path).resolve())
        if self.save_video:
            self.output_video_path = str(Path(self.output_video_path).resolve())
        if self.save_debug_crops:
            self.debug_crops_dir = str(Path(self.debug_crops_dir).resolve())
    
    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    
    def print_config(self):
        """Print configuration in a readable format."""
        print("\n" + "="*60)
        print("PIPELINE CONFIGURATION")
        print("="*60)
        
        sections = {
            "Video Source": ["source", "width", "height", "fps_target"],
            "Detection & Tracking": ["device", "yolo_model", "min_confidence", 
                                     "iou_threshold", "tracker_max_age", "tracker_min_hits"],
            "Attribute Analysis": ["analyze_interval_frames", "min_crop_size", 
                                   "crop_padding", "deepface_backend", "deepface_detector_enforce"],
            "Performance": ["process_every_n", "frame_queue_size", "processing_threads"],
            "Output": ["draw_labels", "show_fps", "show_display", "jsonl_path", 
                      "save_video", "output_video_path"],
            "Debug": ["debug", "save_debug_crops", "debug_crops_dir", "verbose"]
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            for key in keys:
                value = getattr(self, key)
                # Truncate long paths
                if isinstance(value, str) and len(value) > 50:
                    value = "..." + value[-47:]
                print(f"  {key:28s} = {value}")
        
        print("\n" + "="*60 + "\n")
    
    def to_dict(self) -> dict:
        """Export config as dictionary."""
        return asdict(self)
    
    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved to: {path}")


def load_config() -> Config:
    """
    Load configuration from multiple sources.
    
    Priority (highest to lowest):
    1. Command-line arguments
    2. YAML file (via --config)
    3. Environment variables (limited support)
    4. Default values
    
    Returns:
        Config object with validated parameters
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Modular Face Detection + Age/Gender Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument("--config", type=str, 
                       help="Path to YAML configuration file")
    
    # Video source
    parser.add_argument("--source", type=str, 
                       help="Video source (camera index, file path, or RTSP URL)")
    parser.add_argument("--width", type=int, 
                       help="Frame width in pixels")
    parser.add_argument("--height", type=int, 
                       help="Frame height in pixels")
    
    # Detection & tracking
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"],
                       help="Compute device")
    parser.add_argument("--yolo-model", type=str,
                       help="Path to YOLO model weights")
    parser.add_argument("--min-confidence", type=float,
                       help="Minimum detection confidence (0-1)")
    parser.add_argument("--iou-threshold", type=float,
                       help="IoU threshold for NMS")
    
    # Attribute analysis
    parser.add_argument("--analyze-interval", type=int,
                       help="Frames between attribute re-analysis")
    parser.add_argument("--min-crop-size", type=int,
                       help="Minimum face crop size in pixels")
    parser.add_argument("--crop-padding", type=float,
                       help="Padding ratio around face crops")
    parser.add_argument("--deepface-backend", type=str,
                       choices=["skip", "opencv", "retinaface", "mtcnn"],
                       help="DeepFace face detector backend")
    
    # Performance
    parser.add_argument("--process-every-n", type=int,
                       help="Process every Nth frame (frame skipping)")
    
    # Output
    parser.add_argument("--jsonl-path", type=str,
                       help="Path to JSONL log file")
    parser.add_argument("--save-video", action="store_true",
                       help="Save annotated video output")
    parser.add_argument("--output-video-path", type=str,
                       help="Path for output video file")
    parser.add_argument("--no-display", action="store_true",
                       help="Headless mode (disable GUI window)")
    parser.add_argument("--no-labels", action="store_true",
                       help="Disable bounding box labels")
    
    # Debug
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--save-debug-crops", action="store_true",
                       help="Save face crops for debugging")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    # Special actions
    parser.add_argument("--save-config", type=str,
                       help="Save effective config to YAML and exit")
    
    args = parser.parse_args()
    
    # Step 1: Start with defaults
    cfg_dict = {}
    
    # Step 2: Load environment variables (limited support)
    env_mappings = {
        "CV_SOURCE": "source",
        "CV_DEVICE": "device",
        "CV_DEBUG": "debug",
    }
    for env_key, cfg_key in env_mappings.items():
        env_val = os.getenv(env_key)
        if env_val:
            # Convert boolean strings
            if env_val.lower() in ["true", "1", "yes"]:
                cfg_dict[cfg_key] = True
            elif env_val.lower() in ["false", "0", "no"]:
                cfg_dict[cfg_key] = False
            else:
                cfg_dict[cfg_key] = env_val
    
    # Step 3: Load YAML file
    if args.config:
        yaml_path = Path(args.config)
        if not yaml_path.exists():
            print(f"Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        
        try:
            with open(yaml_path, "r") as f:
                yaml_cfg = yaml.safe_load(f)
                if yaml_cfg:
                    cfg_dict.update(yaml_cfg)
                    if args.verbose or os.getenv("CV_VERBOSE"):
                        print(f"Loaded config from: {args.config}")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading config file: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Step 4: Override with CLI arguments (highest priority)
    cli_overrides = {
        "source": args.source,
        "width": args.width,
        "height": args.height,
        "device": args.device,
        "yolo_model": args.yolo_model,
        "min_confidence": args.min_confidence,
        "iou_threshold": args.iou_threshold,
        "analyze_interval_frames": args.analyze_interval,
        "min_crop_size": args.min_crop_size,
        "crop_padding": args.crop_padding,
        "deepface_backend": args.deepface_backend,
        "process_every_n": args.process_every_n,
        "jsonl_path": args.jsonl_path,
        "save_video": args.save_video,
        "output_video_path": args.output_video_path,
        "show_display": not args.no_display if args.no_display else None,
        "draw_labels": not args.no_labels if args.no_labels else None,
        "debug": args.debug if args.debug else None,
        "save_debug_crops": args.save_debug_crops if args.save_debug_crops else None,
        "verbose": args.verbose if args.verbose else None,
    }
    
    # Only override if explicitly provided (not None)
    for key, value in cli_overrides.items():
        if value is not None:
            cfg_dict[key] = value
    
    # Step 5: Create Config object (applies defaults for missing keys)
    try:
        config = Config(**cfg_dict)
    except TypeError as e:
        print(f"Invalid configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Step 6: Ensure required directories exist
    _ensure_directories(config)
    
    # Step 7: Validate YOLO model exists
    if not Path(config.yolo_model).exists():
        print(f"YOLO model not found: {config.yolo_model}", file=sys.stderr)
        print("   Please download yolov8n-face-lindevs.pt to weights/ directory", file=sys.stderr)
        sys.exit(1)
    
    # Step 8: Print configuration if verbose
    if config.verbose or config.debug:
        config.print_config()
    
    # Step 9: Handle --save-config
    if args.save_config:
        config.save_yaml(args.save_config)
        print(f"Configuration saved to: {args.save_config}")
        sys.exit(0)
    
    return config


def _ensure_directories(config: Config):
    """Create necessary directories for outputs."""
    dirs_to_create = []
    
    # JSONL log directory
    jsonl_dir = Path(config.jsonl_path).parent
    if jsonl_dir and str(jsonl_dir) != ".":
        dirs_to_create.append(jsonl_dir)
    
    # Video output directory
    if config.save_video:
        video_dir = Path(config.output_video_path).parent
        if video_dir and str(video_dir) != ".":
            dirs_to_create.append(video_dir)
    
    # Debug crops directory
    if config.save_debug_crops:
        dirs_to_create.append(Path(config.debug_crops_dir))
    
    # Create directories
    for directory in dirs_to_create:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create directory {directory}: {e}", file=sys.stderr)


def get_config() -> Config:
    """
    Convenience function for getting configuration.
    Alias for load_config().
    """
    return load_config()


# Example usage
if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    config.print_config()
    
    # Test export
    print("\n📄 Config as dict:")
    import json
    print(json.dumps(config.to_dict(), indent=2))
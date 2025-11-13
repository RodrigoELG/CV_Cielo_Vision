# src/config.py
import argparse
import yaml
import os
from dataclasses import dataclass

@dataclass
class Config:
    source: str
    width: int
    height: int
    device: str
    analyze_interval_frames: int
    min_confidence: float
    jsonl_path: str
    draw_labels: bool
    save_video: bool
    output_video_path: str

def load_config():
    parser = argparse.ArgumentParser(description="Face-Age/Gender Pipeline")
    parser.add_argument("--config", type=str, help="YAML config file", default=None)
    parser.add_argument("--source", type=str, default="0", help="Camera index or video file/RTSP")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    parser.add_argument("--analyze_interval_frames", type=int, default=120)
    parser.add_argument("--min_confidence", type=float, default=0.35)
    parser.add_argument("--jsonl_path", type=str, default="logs/events.jsonl")
    parser.add_argument("--draw_labels", action="store_true", help="Draw labels on frames")
    parser.add_argument("--save_video", action="store_true", help="Save annotated video")
    parser.add_argument("--output_video_path", type=str, default="results/output.mp4")

    args = parser.parse_args()
    cfg = vars(args)

    # load YAML if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            y = yaml.safe_load(f)
        cfg.update(y)

    return Config(**cfg)

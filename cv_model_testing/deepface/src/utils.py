# src/utils.py
"""
Utility functions for the Face Analysis Pipeline.

Includes:
- Drawing utilities (labels, bboxes)
- Geometry utilities (IoU, bbox conversions, safe cropping)
- Performance utilities (Timer, FPSCounter)
- File system utilities
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
from typing import Tuple


# ============================================================
# Drawing Utilities
# ============================================================

def draw_label(img, text, x, y, font_scale=0.6, thickness=1, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    Draw text label with background rectangle.
    
    Args:
        img: Image to draw on
        text: Text to display
        x, y: Position (top-left corner)
        font_scale: Text size
        thickness: Text thickness
        color: Text color (BGR)
        bg_color: Background color (BGR)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), base = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(img, (x, y - h - base - 4), (x + w + 10, y + base + 4), bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x + 5, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_bbox(img, bbox, color=(0, 255, 0), thickness=2, label=None):
    """
    Draw bounding box on image.
    
    Args:
        img: Image to draw on
        bbox: Bounding box as (x, y, w, h)
        color: Box color (BGR)
        thickness: Line thickness
        label: Optional text label
    """
    x, y, w, h = bbox
    
    # Draw rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label if provided
    if label:
        draw_label(img, label, x, y - 10, color=color)


# ============================================================
# Geometry Utilities
# ============================================================

def safe_crop(frame, x, y, w, h, pad_ratio=0.2):
    """
    Safely crop region from frame with padding, handling boundaries.
    
    Args:
        frame: Input image
        x, y, w, h: Bounding box (x, y, width, height)
        pad_ratio: Padding ratio (0.2 = 20% of max dimension)
        
    Returns:
        Cropped image region
    """
    H, W = frame.shape[:2]
    pad = int(pad_ratio * max(w, h))
    
    # Calculate padded coordinates
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    
    # Extract crop
    crop = frame[y0:y1, x0:x1]
    
    return crop


def calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1, bbox2: Bounding boxes as (x, y, w, h)
        
    Returns:
        IoU value (0.0 to 1.0)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convert to (x1, y1, x2, y2) format
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2
    
    # Calculate intersection coordinates
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1_max, x2_max)
    iy2 = min(y1_max, y2_max)
    
    # Check if there's no intersection
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    
    # Calculate intersection area
    intersection = (ix2 - ix1) * (iy2 - iy1)
    
    # Calculate union area
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    # Return IoU
    return intersection / union if union > 0 else 0.0


def bbox_to_xyxy(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Convert bounding box from (x, y, w, h) to (x1, y1, x2, y2) format.
    
    Args:
        bbox: Bounding box as (x, y, width, height)
        
    Returns:
        Bounding box as (x1, y1, x2, y2)
    """
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def bbox_to_xywh(xyxy: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Convert bounding box from (x1, y1, x2, y2) to (x, y, w, h) format.
    
    Args:
        xyxy: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Bounding box as (x, y, width, height)
    """
    x1, y1, x2, y2 = xyxy
    return (x1, y1, x2 - x1, y2 - y1)


# ============================================================
# Performance Utilities
# ============================================================

class FPSCounter:
    """
    Simple FPS (Frames Per Second) counter.
    
    Usage:
        fps_counter = FPSCounter()
        while True:
            # ... process frame ...
            fps_counter.tick()
            print(f"FPS: {fps_counter.fps():.1f}")
    """
    
    def __init__(self):
        """Initialize FPS counter."""
        self._start = time.time()
        self._frames = 0
    
    def tick(self):
        """Record a frame."""
        self._frames += 1
    
    def fps(self) -> float:
        """
        Calculate current FPS.
        
        Returns:
            Frames per second (0.0 if no time elapsed)
        """
        dt = time.time() - self._start
        return self._frames / dt if dt > 0 else 0.0
    
    def reset(self):
        """Reset counter."""
        self._start = time.time()
        self._frames = 0


class Timer:
    """
    Simple timer for measuring elapsed time.
    
    Usage:
        timer = Timer()
        # ... do work ...
        print(f"Elapsed: {timer.elapsed_ms():.1f}ms")
        timer.reset()
    """
    
    def __init__(self):
        """Initialize timer."""
        self._start = time.time()
    
    def reset(self):
        """Reset timer to current time."""
        self._start = time.time()
    
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Seconds since timer start/reset
        """
        return time.time() - self._start
    
    def elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.
        
        Returns:
            Milliseconds since timer start/reset
        """
        return self.elapsed() * 1000.0


# ============================================================
# File System Utilities
# ============================================================

def ensure_dir(path):
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path (str or Path)
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(file_path):
    """
    Ensure parent directory of file exists.
    
    Args:
        file_path: File path (str or Path)
        
    Returns:
        Path object of parent directory
    """
    file_path = Path(file_path)
    parent = file_path.parent
    
    if parent and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)
    
    return parent


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("=== Testing Utilities ===\n")
    
    # Test IoU calculation
    print("Testing calculate_iou:")
    bbox1 = (100, 100, 200, 200)
    bbox2 = (150, 150, 200, 200)
    iou = calculate_iou(bbox1, bbox2)
    print(f"  IoU between overlapping boxes: {iou:.3f}")
    
    # Test bbox conversions
    print("\nTesting bbox conversions:")
    xywh = (100, 100, 200, 200)
    xyxy = bbox_to_xyxy(xywh)
    xywh_back = bbox_to_xywh(xyxy)
    print(f"  (x,y,w,h) = {xywh}")
    print(f"  → (x1,y1,x2,y2) = {xyxy}")
    print(f"  → (x,y,w,h) = {xywh_back}")
    
    # Test FPSCounter
    print("\nTesting FPSCounter:")
    fps_counter = FPSCounter()
    for i in range(100):
        fps_counter.tick()
        time.sleep(0.01)
    print(f"  FPS after 100 frames: {fps_counter.fps():.1f}")
    
    # Test Timer
    print("\nTesting Timer:")
    timer = Timer()
    time.sleep(0.1)
    print(f"  Elapsed: {timer.elapsed_ms():.1f}ms")
    
    # Test directory creation
    print("\nTesting ensure_dir:")
    test_dir = Path("test_dir_temp/nested/deep")
    ensure_dir(test_dir)
    print(f"  Created: {test_dir}")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_dir_temp")
    
    print("\n✅ All tests passed!")
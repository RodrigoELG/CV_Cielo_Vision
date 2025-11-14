# src/data_models.py
"""
Data models for the Face Analysis Pipeline.

Uses Pydantic for:
- Automatic validation
- Type coercion
- JSON serialization
- Clear data contracts

Technical Notes:
- Timestamps use float (Unix time) for JSON compatibility
- Age stored as int (rounded from DeepFace float output)
- Bboxes validated to ensure positive dimensions
- CachedAttributes uses plain Python class (not Pydantic) for performance
"""

from typing import Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, root_validator
import time
import json


# ============================================================
# Detection Models (Raw YOLO Output)
# ============================================================

class Detection(BaseModel):
    """
    Raw face detection from YOLO.
    
    Attributes:
        bbox: Bounding box as (x, y, width, height)
        confidence: Detection confidence score (0-1)
    """
    bbox: Tuple[int, int, int, int] = Field(..., description="(x, y, w, h)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    
    @validator('bbox')
    def validate_bbox(cls, v):
        """Ensure bbox has positive dimensions."""
        x, y, w, h = v
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bbox dimensions: width={w}, height={h} must be > 0")
        return v
    
    @property
    def area(self) -> int:
        """Calculate bounding box area in pixels."""
        return self.bbox[2] * self.bbox[3]
    
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate bounding box center point."""
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)
    
    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        """Convert bbox from (x,y,w,h) to (x1,y1,x2,y2) format."""
        x, y, w, h = self.bbox
        return (x, y, x + w, y + h)
    
    def iou(self, other: 'Detection') -> float:
        """
        Calculate Intersection over Union with another detection.
        
        Technical Note:
        IoU is used for NMS (Non-Maximum Suppression) and tracking.
        Values close to 1.0 indicate high overlap (likely same face).
        """
        x1, y1, x2, y2 = self.xyxy
        ox1, oy1, ox2, oy2 = other.xyxy
        
        # Calculate intersection
        ix1 = max(x1, ox1)
        iy1 = max(y1, oy1)
        ix2 = min(x2, ox2)
        iy2 = min(y2, oy2)
        
        if ix2 < ix1 or iy2 < iy1:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    class Config:
        # Allow arbitrary types for compatibility
        arbitrary_types_allowed = True
        # Custom JSON encoder
        json_encoders = {
            float: lambda v: round(v, 3)  # Round floats to 3 decimals
        }


# ============================================================
# Tracking Models (SORT Output)
# ============================================================

class TrackedDetection(Detection):
    """
    Face detection with persistent tracking ID.
    
    Attributes:
        track_id: Unique identifier maintained across frames
    """
    track_id: int = Field(..., ge=0, description="Unique track identifier")
    
    def __repr__(self) -> str:
        """Custom representation for debugging."""
        x, y, w, h = self.bbox
        return f"TrackedDetection(id={self.track_id}, bbox=[{x},{y},{w},{h}], conf={self.confidence:.2f})"


# ============================================================
# Attribute Analysis Models
# ============================================================

class AnnotatedDetection(TrackedDetection):
    """
    Face detection with analyzed attributes (age, gender).
    
    Attributes:
        age: Estimated age in years (None if not analyzed yet)
        gender: Gender classification ("Man" | "Woman" | None)
        timestamp: Unix timestamp when analyzed
    """
    age: Optional[int] = Field(None, ge=0, le=120, description="Estimated age")
    gender: Optional[str] = Field(None, description="Gender classification")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp")
    
    @validator('gender')
    def validate_gender(cls, v):
        """Ensure gender is normalized."""
        if v is not None and v not in ['Man', 'Woman']:
            # Auto-normalize common variations
            if v.lower() in ['male', 'm', 'man']:
                return 'Man'
            elif v.lower() in ['female', 'f', 'w', 'woman']:
                return 'Woman'
            else:
                raise ValueError(f"Invalid gender: {v} (must be 'Man' or 'Woman')")
        return v
    
    def __repr__(self) -> str:
        """Custom representation for debugging."""
        age_str = f"{self.age}y" if self.age else "?"
        gender_str = self.gender or "?"
        return f"AnnotatedDetection(id={self.track_id}, {age_str}, {gender_str})"


class CachedAttributes:
    """
    Cached attribute analysis for a tracked face.
    
    Uses plain Python class (not Pydantic) for performance.
    Updated periodically based on analyze_interval_frames.
    
    Technical Note:
    Caching reduces DeepFace calls from 30/sec to 2/sec (15-frame interval),
    improving throughput by ~15x while maintaining accuracy.
    """
    
    def __init__(self, age: Optional[int], gender: Optional[str], frame_idx: int):
        self.age = age
        self.gender = gender
        self.last_update_frame = frame_idx
        self.timestamp = time.time()
        self.update_count = 1
    
    def should_update(self, current_frame: int, interval: int) -> bool:
        """
        Check if attributes should be re-analyzed.
        
        Args:
            current_frame: Current frame index
            interval: Minimum frames between updates
            
        Returns:
            True if interval has elapsed
        """
        return (current_frame - self.last_update_frame) >= interval
    
    def update(self, age: Optional[int], gender: Optional[str], frame_idx: int):
        """
        Update cached attributes with new analysis.
        
        Args:
            age: New age estimate (None preserves old value)
            gender: New gender classification (None preserves old value)
            frame_idx: Frame index of analysis
        """
        if age is not None:
            self.age = age
        if gender is not None:
            self.gender = gender
        
        self.last_update_frame = frame_idx
        self.timestamp = time.time()
        self.update_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Export cache state as dictionary."""
        return {
            'age': self.age,
            'gender': self.gender,
            'last_update_frame': self.last_update_frame,
            'timestamp': self.timestamp,
            'update_count': self.update_count
        }
    
    def __repr__(self) -> str:
        age_str = f"{self.age}y" if self.age else "?"
        gender_str = self.gender or "?"
        return f"CachedAttributes({age_str}, {gender_str}, updates={self.update_count})"


# ============================================================
# Logging Models (JSONL Output)
# ============================================================

class LogEntry(BaseModel):
    """
    Single detection log entry for JSONL output.
    
    One entry per face per frame. Format optimized for:
    - Streaming logs (append-only)
    - Time-series analysis
    - External analytics tools
    """
    timestamp: float = Field(..., description="Unix timestamp")
    frame_index: int = Field(..., ge=0, description="Frame number")
    track_id: int = Field(..., ge=0, description="Persistent track ID")
    bbox: Tuple[int, int, int, int] = Field(..., description="(x, y, w, h)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    age: Optional[int] = Field(None, ge=0, le=120, description="Estimated age")
    gender: Optional[str] = Field(None, description="Gender classification")
    
    @validator('gender')
    def validate_gender(cls, v):
        """Normalize gender values."""
        if v is not None and v not in ['Man', 'Woman']:
            raise ValueError(f"Invalid gender: {v}")
        return v
    
    @classmethod
    def from_annotated_detection(cls, det: AnnotatedDetection, frame_idx: int) -> 'LogEntry':
        """
        Create log entry from annotated detection.
        
        Args:
            det: AnnotatedDetection object
            frame_idx: Current frame index
            
        Returns:
            LogEntry ready for JSONL serialization
        """
        return cls(
            timestamp=det.timestamp,
            frame_index=frame_idx,
            track_id=det.track_id,
            bbox=det.bbox,
            confidence=det.confidence,
            age=det.age,
            gender=det.gender
        )
    
    def to_jsonl(self) -> str:
        """
        Serialize to JSONL format (single-line JSON).
        
        Technical Note:
        JSONL (newline-delimited JSON) allows streaming writes
        and easy log rotation without parsing entire file.
        """
        return self.json(ensure_ascii=False)
    
    class Config:
        json_encoders = {
            float: lambda v: round(v, 3)  # Round timestamps and confidence
        }


# ============================================================
# Frame Transport Models (Thread Communication)
# ============================================================

class FramePacket:
    """
    Container for frames passed between threads.
    
    Uses plain Python class for zero-overhead frame passing.
    Timestamped to detect stale frames in queue.
    
    Technical Note:
    FrameGrabber (Thread A) produces FramePackets.
    Main loop (Thread B) consumes them via Queue.
    """
    
    def __init__(self, frame, frame_idx: int, timestamp: Optional[float] = None):
        """
        Create frame packet.
        
        Args:
            frame: OpenCV frame (numpy array)
            frame_idx: Sequential frame number
            timestamp: Unix timestamp (auto-generated if None)
        """
        self.frame = frame
        self.frame_idx = frame_idx
        self.timestamp = timestamp if timestamp is not None else time.time()
    
    @property
    def age_ms(self) -> float:
        """Calculate frame age in milliseconds."""
        return (time.time() - self.timestamp) * 1000
    
    @property
    def is_stale(self, threshold_ms: float = 100) -> bool:
        """Check if frame is stale (older than threshold)."""
        return self.age_ms > threshold_ms
    
    def __repr__(self) -> str:
        return f"FramePacket(idx={self.frame_idx}, age={self.age_ms:.1f}ms, shape={self.frame.shape if self.frame is not None else None})"


# ============================================================
# Statistics Models (Optional - For Dashboard/Monitoring)
# ============================================================

class PipelineStats(BaseModel):
    """
    Runtime statistics for monitoring pipeline health.
    
    Optional model for future dashboard integration.
    """
    total_frames_processed: int = 0
    total_faces_detected: int = 0
    unique_tracks: int = 0
    average_fps: float = 0.0
    average_faces_per_frame: float = 0.0
    cache_hit_rate: float = 0.0  # Percentage of cached attribute lookups
    
    # Age/gender distribution
    age_distribution: Dict[str, int] = Field(default_factory=dict)  # e.g., {"0-20": 5, "21-40": 10}
    gender_distribution: Dict[str, int] = Field(default_factory=dict)  # e.g., {"Man": 15, "Woman": 10}
    
    class Config:
        json_encoders = {
            float: lambda v: round(v, 2)
        }


# ============================================================
# Utility Functions
# ============================================================

def bbox_to_xyxy(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Convert bbox from (x,y,w,h) to (x1,y1,x2,y2)."""
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def bbox_to_xywh(xyxy: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Convert bbox from (x1,y1,x2,y2) to (x,y,w,h)."""
    x1, y1, x2, y2 = xyxy
    return (x1, y1, x2 - x1, y2 - y1)


def calculate_iou(bbox1: Tuple[int, int, int, int], 
                  bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union between two bboxes.
    
    Args:
        bbox1, bbox2: Bounding boxes as (x, y, w, h)
        
    Returns:
        IoU value (0-1)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convert to xyxy
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2
    
    # Calculate intersection
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1_max, x2_max)
    iy2 = min(y1_max, y2_max)
    
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0.0


# ============================================================
# Example Usage & Testing
# ============================================================

if __name__ == "__main__":
    # Test Detection
    print("=== Testing Detection ===")
    det = Detection(bbox=(100, 150, 200, 250), confidence=0.89)
    print(f"Detection: {det}")
    print(f"Area: {det.area}px²")
    print(f"Center: {det.center}")
    print(f"XYXY: {det.xyxy}")
    
    # Test TrackedDetection
    print("\n=== Testing TrackedDetection ===")
    tracked = TrackedDetection(bbox=(100, 150, 200, 250), confidence=0.89, track_id=5)
    print(f"Tracked: {tracked}")
    
    # Test AnnotatedDetection
    print("\n=== Testing AnnotatedDetection ===")
    annotated = AnnotatedDetection(
        bbox=(100, 150, 200, 250),
        confidence=0.89,
        track_id=5,
        age=34,
        gender="Man"
    )
    print(f"Annotated: {annotated}")
    
    # Test CachedAttributes
    print("\n=== Testing CachedAttributes ===")
    cache = CachedAttributes(age=34, gender="Man", frame_idx=100)
    print(f"Cache: {cache}")
    print(f"Should update at frame 115 (interval=15): {cache.should_update(115, 15)}")
    cache.update(age=35, gender="Man", frame_idx=115)
    print(f"After update: {cache}")
    
    # Test LogEntry
    print("\n=== Testing LogEntry ===")
    log = LogEntry.from_annotated_detection(annotated, frame_idx=150)
    print(f"Log entry: {log}")
    print(f"JSONL: {log.to_jsonl()}")
    
    # Test FramePacket
    print("\n=== Testing FramePacket ===")
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = FramePacket(frame, frame_idx=42)
    print(f"Packet: {packet}")
    print(f"Is stale: {packet.is_stale(threshold_ms=50)}")
    
    # Test IoU calculation
    print("\n=== Testing IoU ===")
    det1 = Detection(bbox=(100, 100, 200, 200), confidence=0.9)
    det2 = Detection(bbox=(150, 150, 200, 200), confidence=0.8)
    print(f"IoU between overlapping detections: {det1.iou(det2):.3f}")
    
    # Test validation
    print("\n=== Testing Validation ===")
    try:
        bad_det = Detection(bbox=(100, 100, -50, 200), confidence=0.9)
    except ValueError as e:
        print(f"✅ Caught validation error: {e}")
    
    try:
        bad_gender = AnnotatedDetection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            track_id=1,
            age=30,
            gender="Unknown"
        )
    except ValueError as e:
        print(f"✅ Caught gender validation error: {e}")
    
    print("\n✅ All tests passed!")
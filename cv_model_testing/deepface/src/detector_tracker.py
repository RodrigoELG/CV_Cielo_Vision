# src/detector_tracker.py
"""
Face detection and tracking module.

Combines:
- YOLOv8: State-of-the-art object detection (face detection variant)
- SORT/ByteTrack: Multi-object tracking with persistent IDs

Technical Notes:
- YOLOv8 provides bounding boxes + confidence scores
- Tracking maintains IDs across frames (survives occlusions)
- NMS (Non-Maximum Suppression) filters duplicate detections
- Track IDs enable attribute caching in analyzer
- GPU acceleration via PyTorch (10x faster than CPU)

Performance:
- Detection: ~8-12ms per frame (GPU)
- Tracking: ~2-3ms per frame
- Total: ~10-15ms = 60-100 FPS capability

Architecture:
    Input Frame → YOLO Detection → NMS → SORT Tracking → TrackedDetections
"""

import numpy as np
import sys
import cv2
from typing import List, Optional, Tuple
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed", file=sys.stderr)
    print("   Install with: pip install ultralytics", file=sys.stderr)
    sys.exit(1)

from .data_models import TrackedDetection
try:
    from .utils import calculate_iou
except ImportError:
    pass # Ignore for now; utility functions may be defined later

class FaceDetectorTracker:
    """
    YOLOv8 Face Detection + SORT/ByteTrack Tracking.
    
    Features:
    - GPU-accelerated detection
    - Persistent track IDs
    - Configurable confidence/IoU thresholds
    - Built-in NMS
    - Track state management
    
    Usage:
        detector = FaceDetectorTracker("weights/yolov8n-face.pt", device="cuda")
        tracked = detector.detect_and_track(frame)
        for det in tracked:
            print(f"Track ID: {det.track_id}, Bbox: {det.bbox}")
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        min_confidence: float = 0.35,
        iou_threshold: float = 0.5,
        tracker_type: str = "bytetrack",
        max_age: int = 30,
        min_hits: int = 3,
        verbose: bool = False
    ):
        """
        Initialize face detector and tracker.
        
        Args:
            model_path: Path to YOLOv8 weights file (.pt)
            device: 'cuda' or 'cpu'
            min_confidence: Minimum detection confidence (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
            tracker_type: 'bytetrack' or 'botsort'
            max_age: Max frames to keep track without detection
            min_hits: Min detections before track confirmation
            verbose: Enable detailed logging
        
        Technical Notes:
        - min_confidence: Higher = fewer false positives, lower = more detections
        - iou_threshold: Lower = stricter NMS (removes more overlaps)
        - max_age: Higher = tracks survive longer occlusions
        - min_hits: Higher = fewer false tracks, but slower initialization
        """
        self.model_path = str(Path(model_path).resolve())
        self.device = device
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold
        self.tracker_type = tracker_type
        self.max_age = max_age
        self.min_hits = min_hits
        self.verbose = verbose
        
        # Statistics
        self.total_detections = 0
        self.total_tracks_created = 0
        self.active_tracks = 0
        
        # Load model
        self._load_model()
        
        # Configure tracker
        self._configure_tracker()
    
    def _load_model(self):
        """Load YOLOv8 model with error handling."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        
        if self.verbose:
            print(f"[Detector] Loading YOLOv8 model: {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            
            # Move model to device
            if self.device == "cuda":
                try:
                    self.model.to("cuda")
                    if self.verbose:
                        print(f"[Detector] Model loaded on GPU")
                except Exception as e:
                    print(f"[Detector] GPU unavailable, falling back to CPU: {e}")
                    self.device = "cpu"
                    self.model.to("cpu")
            else:
                self.model.to("cpu")
                if self.verbose:
                    print(f"[Detector] Model loaded on CPU")
        
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def _configure_tracker(self):
        """Configure tracking parameters."""
        self.tracker_config = {
            "tracker_type": self.tracker_type,
            "track_high_thresh": self.min_confidence,
            "track_low_thresh": self.min_confidence * 0.5,
            "new_track_thresh": self.min_confidence * 0.6,
            "track_buffer": self.max_age,
            "match_thresh": 0.8,
            "min_hits": self.min_hits,
        }
        
        if self.verbose:
            print(f"[Detector] Tracker: {self.tracker_type}")
            print(f"[Detector] Confidence threshold: {self.min_confidence}")
            print(f"[Detector] IoU threshold: {self.iou_threshold}")
            print(f"[Detector] Max age: {self.max_age} frames")
            print(f"[Detector] Min hits: {self.min_hits} detections")
    
    def detect_and_track(self, frame: np.ndarray) -> List[TrackedDetection]:
        """
        Run detection and tracking on a frame.
        
        Args:
            frame: BGR image from OpenCV (H, W, 3)
            
        Returns:
            List of TrackedDetection objects with persistent track_id
        
        Technical Flow:
        1. YOLOv8 inference (detects faces)
        2. NMS filtering (removes duplicates)
        3. Confidence filtering (removes low-confidence)
        4. SORT tracking (assigns/updates track IDs)
        5. Convert to TrackedDetection objects
        """
        if frame is None or frame.size == 0:
            return []
        
        # Run YOLO with tracking
        # persist=True maintains track IDs across frames
        try:
            results = self.model.track(
                source=frame,
                conf=self.min_confidence,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                persist=True,  # Critical: enables persistent tracking
                tracker=f"{self.tracker_type}.yaml"
            )
        except Exception as e:
            if self.verbose:
                print(f"[Detector] Detection failed: {e}")
            return []
        
        # Parse results
        tracked_detections = self._parse_results(results)
        
        # Update statistics
        self.total_detections += len(tracked_detections)
        self.active_tracks = len(set(d.track_id for d in tracked_detections))
        
        return tracked_detections
    
    def _parse_results(self, results) -> List[TrackedDetection]:
        """
        Parse YOLO results into TrackedDetection objects.
        
        Args:
            results: YOLO results object
            
        Returns:
            List of TrackedDetection objects
        """
        tracked_detections = []
        
        if not results or len(results) == 0:
            return tracked_detections
        
        result = results[0]  # Single frame result
        
        # Check if boxes exist
        if not hasattr(result, 'boxes') or result.boxes is None:
            return tracked_detections
        
        boxes = result.boxes
        
        # Check if we have any detections
        if len(boxes) == 0:
            return tracked_detections
        
        # Extract tracking IDs (if available)
        if hasattr(boxes, 'id') and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            # Fallback: use detection index as ID (not persistent!)
            track_ids = np.arange(len(boxes))
            if self.verbose:
                print("[Detector] Warning: Track IDs not available, using indices")
        
        # Extract bounding boxes and confidences
        try:
            xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
            confs = boxes.conf.cpu().numpy()
        except Exception as e:
            if self.verbose:
                print(f"[Detector] Failed to extract boxes: {e}")
            return tracked_detections
        
        # Convert to TrackedDetection objects
        for (x1, y1, x2, y2), conf, track_id in zip(xyxy, confs, track_ids):
            # Convert to (x, y, w, h) format
            x = int(max(0, x1))
            y = int(max(0, y1))
            w = int(max(0, x2 - x1))
            h = int(max(0, y2 - y1))
            
            # Validate bbox
            if w <= 0 or h <= 0:
                continue
            
            # Create TrackedDetection
            try:
                detection = TrackedDetection(
                    bbox=(x, y, w, h),
                    confidence=float(conf),
                    track_id=int(track_id)
                )
                tracked_detections.append(detection)
                
                # Track new IDs
                if track_id not in self._seen_track_ids:
                    self._seen_track_ids.add(track_id)
                    self.total_tracks_created += 1
            
            except Exception as e:
                if self.verbose:
                    print(f"[Detector] Failed to create TrackedDetection: {e}")
                continue
        
        return tracked_detections
    
    @property
    def _seen_track_ids(self):
        """Track IDs seen so far (for statistics)."""
        if not hasattr(self, '_track_ids_set'):
            self._track_ids_set = set()
        return self._track_ids_set
    
    def reset_tracker(self):
        """
        Reset tracker state.
        
        Useful when:
        - Video source changes
        - Long pause in processing
        - Tracking quality degrades
        """
        if self.verbose:
            print("[Detector] Resetting tracker state")
        
        # Ultralytics resets automatically with persist=True
        # But we clear our statistics
        self.total_detections = 0
        self.total_tracks_created = 0
        self.active_tracks = 0
        if hasattr(self, '_track_ids_set'):
            self._track_ids_set.clear()
    
    def get_stats(self) -> dict:
        """Get detection and tracking statistics."""
        return {
            'total_detections': self.total_detections,
            'total_tracks_created': self.total_tracks_created,
            'active_tracks': self.active_tracks,
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.min_confidence,
            'iou_threshold': self.iou_threshold,
            'tracker_type': self.tracker_type,
        }
    
    def print_stats(self):
        """Print detection and tracking statistics."""
        stats = self.get_stats()
        print("\n=== Detector/Tracker Statistics ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key:25s}: {value:.3f}")
            else:
                print(f"  {key:25s}: {value}")
        print("=" * 40 + "\n")
    
    def warmup(self, size: Tuple[int, int] = (640, 480)):
        """
        Warm up model with dummy inference.
        
        First inference is slow due to:
        - CUDA initialization
        - Model compilation
        - Memory allocation
        
        Warmup ensures first real frame is fast.
        
        Args:
            size: Frame size (width, height)
        """
        if self.verbose:
            print("[Detector] Warming up model...")
        
        # Create dummy frame
        dummy_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Run inference
        try:
            _ = self.detect_and_track(dummy_frame)
            if self.verbose:
                print("[Detector] Warmup complete")
        except Exception as e:
            print(f"[Detector] Warmup failed: {e}")


# ============================================================
# Utility Functions
# ============================================================

def test_detector():
    """Test detector with sample data."""
    import cv2
    import time
    
    print("=== Testing FaceDetectorTracker ===\n")
    
    # Check if model exists
    model_path = "weights/yolov8n-face-lindevs.pt"
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("   Please download the model first")
        return
    
    # Create detector
    print("Creating detector...")
    detector = FaceDetectorTracker(
        model_path=model_path,
        device="cpu",  # Use CPU for testing
        verbose=True
    )
    
    # Warmup
    detector.warmup()
    
    # Create test frame (blank with white rectangle simulating face)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (400, 400), (255, 255, 255), -1)
    
    # Test detection
    print("\nTesting detection...")
    start = time.time()
    detections = detector.detect_and_track(frame)
    elapsed = (time.time() - start) * 1000
    
    print(f"Detections: {len(detections)}")
    print(f"Time: {elapsed:.1f}ms")
    
    for det in detections:
        print(f"  - Track ID: {det.track_id}, Bbox: {det.bbox}, Conf: {det.confidence:.2f}")
    
    # Test on multiple frames (simulating tracking)
    print("\nTesting tracking (10 frames)...")
    for i in range(10):
        detections = detector.detect_and_track(frame)
        if detections:
            print(f"Frame {i}: Track IDs = {[d.track_id for d in detections]}")
    
    # Print stats
    detector.print_stats()
    
    print("Test complete")


def benchmark_detector(model_path: str, device: str = "cpu", frames: int = 100):
    """
    Benchmark detector performance.
    
    Args:
        model_path: Path to YOLO model
        device: 'cpu' or 'cuda'
        frames: Number of frames to test
    """
    import time
    
    print(f"=== Benchmarking Detector ({device.upper()}) ===\n")
    
    # Create detector
    detector = FaceDetectorTracker(
        model_path=model_path,
        device=device,
        verbose=False
    )
    
    # Warmup
    print("Warming up...")
    detector.warmup()
    
    # Create test frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (300, 200), (600, 600), (255, 255, 255), -1)
    
    # Benchmark
    print(f"Running {frames} frames...")
    times = []
    
    for i in range(frames):
        start = time.time()
        detections = detector.detect_and_track(frame)
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # Convert to ms
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{frames} frames")
    
    # Calculate statistics
    times = np.array(times)
    
    print("\n=== Results ===")
    print(f"Mean:   {times.mean():.2f}ms")
    print(f"Median: {np.median(times):.2f}ms")
    print(f"Min:    {times.min():.2f}ms")
    print(f"Max:    {times.max():.2f}ms")
    print(f"P95:    {np.percentile(times, 95):.2f}ms")
    print(f"P99:    {np.percentile(times, 99):.2f}ms")
    print(f"\nFPS:    {1000 / times.mean():.1f}")
    
    # Print stats
    detector.print_stats()


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Benchmark mode
        model_path = "weights/yolov8n-face-lindevs.pt"
        device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
        benchmark_detector(model_path, device=device, frames=100)
    else:
        # Test mode
        test_detector()
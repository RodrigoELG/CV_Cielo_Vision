# src/attributes.py
"""
Attribute analysis module for age and gender estimation.

Uses DeepFace with caching to minimize redundant inference calls.

Technical Notes:
- BGR→RGB conversion critical for model accuracy
- Caching reduces calls from 30/sec to 2/sec (15-frame interval)
- Minimum crop size (80×80px) ensures quality predictions
- Padding (25%) provides facial context for better accuracy
- Gender normalization: DeepFace outputs vary, we standardize to "Man"/"Woman"
- Age rounding: DeepFace returns float (34.7), we convert to int (35)

Performance:
- Without cache: ~80ms per face @ 30fps = bottleneck
- With cache: ~5ms per face @ 30fps = real-time capable
"""

import os
import cv2
import numpy as np
import tempfile
import time
from typing import List, Optional, Tuple, Dict
from deepface import DeepFace

from .data_models import TrackedDetection, AnnotatedDetection, CachedAttributes
from .utils import safe_crop


class AttributeAnalyzer:
    """
    Analyze age and gender attributes with intelligent caching.
    
    Architecture:
    1. Check cache for track_id
    2. If stale or missing, extract & validate crop
    3. Convert BGR→RGB (critical for accuracy)
    4. Call DeepFace.analyze()
    5. Parse & normalize results
    6. Update cache
    7. Return AnnotatedDetection
    """
    
    def __init__(
        self,
        analyze_interval_frames: int = 15,
        min_crop_size: int = 80,
        crop_padding: float = 0.25,
        backend: str = "skip",
        enforce_detection: bool = False,
        actions: List[str] = None,
        debug: bool = False,
        save_debug_crops: bool = False,
        debug_crops_dir: str = "debug/crops",
        verbose: bool = False
    ):
        """
        Initialize attribute analyzer.
        
        Args:
            analyze_interval_frames: Re-analyze attributes every N frames
            min_crop_size: Minimum face dimension in pixels
            crop_padding: Padding ratio around bbox (0.25 = 25%)
            backend: DeepFace detector backend ('skip' = use our crops)
            enforce_detection: Force DeepFace to find faces in crop
            actions: DeepFace actions to perform (default: ['age', 'gender'])
            debug: Enable debug logging
            save_debug_crops: Save face crops for inspection
            debug_crops_dir: Directory for debug crops
            verbose: Detailed logging
        """
        self.interval = analyze_interval_frames
        self.min_crop_size = min_crop_size
        self.crop_padding = crop_padding
        self.backend = backend
        self.enforce_detection = enforce_detection
        self.actions = actions or ['age', 'gender']
        self.debug = debug
        self.save_debug_crops = save_debug_crops
        self.debug_crops_dir = debug_crops_dir
        self.verbose = verbose
        
        # Cache: track_id → CachedAttributes
        self.cache: Dict[int, CachedAttributes] = {}
        
        # Statistics
        self.total_analyses = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.failed_analyses = 0
        
        # Pre-warm models (reduces first-call latency)
        self._warmup_models()
        
        # Create debug directory if needed
        if save_debug_crops:
            os.makedirs(debug_crops_dir, exist_ok=True)
            if verbose:
                print(f"[AttributeAnalyzer] Debug crops will be saved to: {debug_crops_dir}")
    
    def _warmup_models(self):
        """
        Pre-load DeepFace models to avoid first-call delay.
        
        Technical Note:
        First DeepFace call downloads models (~100MB) and loads into memory.
        This can take 2-3 seconds, blocking the pipeline.
        Pre-warming ensures models are ready before processing starts.
        """
        if self.verbose:
            print("[AttributeAnalyzer] Pre-warming DeepFace models...")
        
        try:
            # Load models explicitly
            self.age_model = DeepFace.build_model(model_name='Age')
            self.gender_model = DeepFace.build_model(model_name='Gender')
            
            if self.verbose:
                print(f"[AttributeAnalyzer] Age model loaded: {type(self.age_model).__name__}")
                print(f"[AttributeAnalyzer] Gender model loaded: {type(self.gender_model).__name__}")
        
        except Exception as e:
            print(f"[AttributeAnalyzer] Warning: Failed to pre-load models: {e}")
            self.age_model = None
            self.gender_model = None
    
    def analyze(
        self,
        tracked_list: List[TrackedDetection],
        frame: np.ndarray,
        frame_index: int
    ) -> List[AnnotatedDetection]:
        """
        Analyze attributes for tracked detections.
        
        Args:
            tracked_list: List of tracked face detections
            frame: BGR frame from OpenCV
            frame_index: Current frame number
            
        Returns:
            List of annotated detections with age/gender
        """
        annotated = []
        
        for td in tracked_list:
            track_id = td.track_id
            
            # Check cache
            cached = self.cache.get(track_id)
            should_analyze = self._should_analyze(cached, frame_index)
            
            age = None
            gender = None
            
            if should_analyze:
                # Cache miss or stale
                self.cache_misses += 1
                
                # Extract and analyze crop
                age, gender = self._analyze_face(td, frame, track_id, frame_index)
                
                # Update cache
                if cached is None:
                    self.cache[track_id] = CachedAttributes(age, gender, frame_index)
                else:
                    cached.update(age, gender, frame_index)
            else:
                # Cache hit
                self.cache_hits += 1
                age = cached.age
                gender = cached.gender
            
            # Create annotated detection
            annotated.append(
                AnnotatedDetection(
                    bbox=td.bbox,
                    confidence=td.confidence,
                    track_id=track_id,
                    age=age,
                    gender=gender,
                    timestamp=time.time()
                )
            )
        
        return annotated
    
    def _should_analyze(self, cached: Optional[CachedAttributes], frame_index: int) -> bool:
        """Determine if analysis is needed."""
        if cached is None:
            return True  # First time seeing this track
        
        return cached.should_update(frame_index, self.interval)
    
    def _analyze_face(
        self,
        detection: TrackedDetection,
        frame: np.ndarray,
        track_id: int,
        frame_index: int
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Analyze a single face detection.
        
        Returns:
            (age, gender) tuple or (None, None) on failure
        """
        self.total_analyses += 1
        
        # Extract bbox (format: x, y, w, h)
        x, y, w, h = detection.bbox
        
        # Validate bbox
        if w <= 0 or h <= 0:
            if self.debug:
                print(f"[AttributeAnalyzer] Invalid bbox for track {track_id}: w={w}, h={h}")
            return None, None
        
        # Extract crop with padding
        crop_bgr = safe_crop(
            frame,
            x, y, w, h,
            pad_ratio=self.crop_padding
        )
        
        # Validate crop
        if not self._is_valid_crop(crop_bgr, track_id):
            return None, None
        
        # Save debug crop if enabled
        if self.save_debug_crops:
            self._save_debug_crop(crop_bgr, track_id, frame_index)
        
        # CRITICAL: Convert BGR → RGB
        # OpenCV uses BGR, DeepFace/TensorFlow expects RGB
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        # Ensure correct dtype and memory layout
        crop_rgb = np.ascontiguousarray(crop_rgb, dtype=np.uint8)
        
        # Attempt analysis
        result = self._deepface_analyze(crop_rgb, track_id)
        
        if result is None:
            # Try fallback: save to temp file
            result = self._deepface_analyze_file(crop_rgb, track_id)
        
        if result is None:
            self.failed_analyses += 1
            return None, None
        
        # Parse result
        age, gender = self._parse_result(result)
        
        if self.verbose and age is not None:
            print(f"[AttributeAnalyzer] Track {track_id}: age={age}, gender={gender}")
        
        return age, gender
    
    def _is_valid_crop(self, crop: np.ndarray, track_id: int) -> bool:
        """Validate crop dimensions and content."""
        if crop is None or crop.size == 0:
            if self.debug:
                print(f"[AttributeAnalyzer] Empty crop for track {track_id}")
            return False
        
        h, w = crop.shape[:2]
        
        if h < self.min_crop_size or w < self.min_crop_size:
            if self.debug:
                print(f"[AttributeAnalyzer] Crop too small for track {track_id}: {w}×{h} (min: {self.min_crop_size})")
            return False
        
        return True
    
    def _save_debug_crop(self, crop: np.ndarray, track_id: int, frame_index: int):
        """Save face crop for debugging."""
        try:
            filename = f"track_{track_id:04d}_frame_{frame_index:06d}.jpg"
            path = os.path.join(self.debug_crops_dir, filename)
            cv2.imwrite(path, crop)
        except Exception as e:
            if self.debug:
                print(f"[AttributeAnalyzer] Failed to save debug crop: {e}")
    
    def _deepface_analyze(
        self,
        crop_rgb: np.ndarray,
        track_id: int
    ) -> Optional[dict]:
        """
        Analyze crop with DeepFace (numpy array input).
        
        Technical Note:
        Some DeepFace versions accept numpy arrays directly.
        This is faster than file-based approach.
        """
        try:
            # Build models dict if available
            models = {}
            if hasattr(self, 'age_model') and self.age_model is not None:
                models["age"] = self.age_model
            if hasattr(self, 'gender_model') and self.gender_model is not None:
                models["gender"] = self.gender_model
            
            # Call DeepFace
            result = DeepFace.analyze(
                img_path=crop_rgb,
                actions=self.actions,
                detector_backend=self.backend,  # 'skip' = no face detection
                enforce_detection=self.enforce_detection,
                prog_bar=False,
                silent=True
            )
            
            return result
        
        except Exception as e:
            if self.debug:
                print(f"[AttributeAnalyzer] DeepFace analysis failed for track {track_id}: {e}")
            return None
    
    def _deepface_analyze_file(
        self,
        crop_rgb: np.ndarray,
        track_id: int
    ) -> Optional[dict]:
        """
        Analyze crop with DeepFace (file-based fallback).
        
        Technical Note:
        Some DeepFace versions or backends require file input.
        This is slower but more compatible.
        """
        tmp_path = None
        
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                # Convert RGB back to BGR for cv2.imwrite
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(tmp_path, crop_bgr)
            
            # Build models dict
            models = {}
            if hasattr(self, 'age_model') and self.age_model is not None:
                models["age"] = self.age_model
            if hasattr(self, 'gender_model') and self.gender_model is not None:
                models["gender"] = self.gender_model
            
            # Call DeepFace with file path
            result = DeepFace.analyze(
                img_path=tmp_path,
                actions=self.actions,
                detector_backend=self.backend,
                enforce_detection=self.enforce_detection,
                prog_bar=False,
                silent=True
            )
            
            return result
        
        except Exception as e:
            if self.debug:
                print(f"[AttributeAnalyzer] File-based analysis failed for track {track_id}: {e}")
            return None
        
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
    
    def _parse_result(self, result) -> Tuple[Optional[int], Optional[str]]:
        """
        Parse DeepFace result to extract age and gender.
        
        DeepFace returns different formats depending on version:
        - Single dict: {'age': 25.3, 'gender': {'Man': 0.99, 'Woman': 0.01}}
        - List of dicts: [{'age': 25.3, 'dominant_gender': 'Man', ...}]
        
        We normalize to: (age: int, gender: "Man"|"Woman"|None)
        """
        # Handle list response (multi-face detection)
        if isinstance(result, list):
            if len(result) == 0:
                return None, None
            result = result[0]  # Take first face
        
        if not isinstance(result, dict):
            return None, None
        
        # Extract age
        age = result.get('age')
        if age is not None:
            try:
                # Convert float to int (round to nearest year)
                age = int(round(float(age)))
                
                # Sanity check
                if age < 0 or age > 120:
                    if self.debug:
                        print(f"[AttributeAnalyzer] Unrealistic age: {age}")
                    age = None
            except (ValueError, TypeError):
                age = None
        
        # Extract gender (multiple possible formats)
        gender = None
        
        # Format 1: 'dominant_gender' key
        if 'dominant_gender' in result:
            gender = result['dominant_gender']
        
        # Format 2: 'gender' as string
        elif 'gender' in result and isinstance(result['gender'], str):
            gender = result['gender']
        
        # Format 3: 'gender' as dict with probabilities
        elif 'gender' in result and isinstance(result['gender'], dict):
            gender_dict = result['gender']
            if gender_dict:
                # Take highest probability
                gender = max(gender_dict, key=gender_dict.get)
        
        # Normalize gender labels
        if gender:
            gender = self._normalize_gender(gender)
        
        return age, gender
    
    def _normalize_gender(self, gender: str) -> Optional[str]:
        """
        Normalize gender labels to standard format.
        
        DeepFace variations: "Man", "Male", "M", "Woman", "Female", "F"
        Our standard: "Man" | "Woman"
        """
        gender_lower = gender.lower().strip()
        
        # Male variations
        if gender_lower in ['man', 'male', 'm']:
            return 'Man'
        
        # Female variations
        if gender_lower in ['woman', 'female', 'f', 'w']:
            return 'Woman'
        
        # Unknown
        if self.debug:
            print(f"[AttributeAnalyzer] Unknown gender label: {gender}")
        
        return None
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """
        Remove cache entries for tracks no longer active.
        
        Prevents memory leak in long-running sessions.
        """
        stale_ids = [tid for tid in self.cache.keys() if tid not in active_track_ids]
        
        if stale_ids and self.verbose:
            print(f"[AttributeAnalyzer] Cleaning up {len(stale_ids)} stale tracks")
        
        for tid in stale_ids:
            del self.cache[tid]
    
    def get_stats(self) -> dict:
        """Get analysis statistics."""
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0.0
        
        return {
            'total_analyses': self.total_analyses,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'failed_analyses': self.failed_analyses,
            'active_tracks': len(self.cache),
        }
    
    def print_stats(self):
        """Print analysis statistics."""
        stats = self.get_stats()
        print("\n=== Attribute Analyzer Statistics ===")
        for key, value in stats.items():
            print(f"  {key:20s}: {value}")
        print("=" * 40 + "\n")


# ============================================================
# Utility Functions for Testing
# ============================================================

def test_analyzer():
    """Test attribute analyzer with sample data."""
    import numpy as np
    from .data_models import TrackedDetection
    
    print("=== Testing AttributeAnalyzer ===\n")
    
    # Create analyzer
    analyzer = AttributeAnalyzer(
        analyze_interval_frames=15,
        debug=True,
        verbose=True
    )
    
    # Create fake frame (black with white rectangle for face)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (400, 400), (255, 255, 255), -1)
    
    # Create fake detection
    detection = TrackedDetection(
        bbox=(200, 150, 200, 250),
        confidence=0.9,
        track_id=1
    )
    
    # Analyze
    print("Analyzing frame 0...")
    results = analyzer.analyze([detection], frame, frame_index=0)
    
    print(f"\nResult: {results[0]}")
    
    # Test cache hit
    print("\nAnalyzing frame 5 (should use cache)...")
    results = analyzer.analyze([detection], frame, frame_index=5)
    
    # Test cache miss (interval reached)
    print("\nAnalyzing frame 15 (should re-analyze)...")
    results = analyzer.analyze([detection], frame, frame_index=15)
    
    # Print stats
    analyzer.print_stats()
    
    print("Test complete")


if __name__ == "__main__":
    test_analyzer()
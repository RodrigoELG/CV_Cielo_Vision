# src/renderer_logger.py
"""
Rendering and logging module for pipeline visualization and data persistence.

Responsibilities:
1. Draw bounding boxes and labels on frames
2. Display FPS and statistics overlay
3. Write JSONL logs for analytics
4. Optionally save annotated video
5. Handle keyboard input for user control

Technical Notes:
- JSONL format: one JSON object per line (streaming-friendly)
- Buffering=1 ensures immediate writes (critical for crashes)
- Logs only when attributes change (avoids 30x duplicate entries)
- Headless mode support for servers without display
- FPS smoothing with exponential moving average

Performance:
- Drawing overhead: ~2ms per frame
- JSONL write: ~0.5ms per entry
- Video encoding: ~5-10ms per frame (depends on codec)
"""

import cv2
import json
import time
import sys
from typing import List, Optional
from pathlib import Path

from .data_models import AnnotatedDetection, LogEntry
from .utils import draw_label


class RendererLogger:
    """
    Handles frame rendering, logging, and video output.
    
    Features:
    - Bounding box + label rendering
    - Real-time FPS display
    - Statistics overlay (face count, age distribution)
    - JSONL logging (de-duplicated)
    - Video recording
    - Headless mode support
    """
    
    def __init__(
        self,
        jsonl_path: str,
        draw_labels: bool = True,
        show_fps: bool = True,
        show_display: bool = True,
        save_video: bool = False,
        output_video_path: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        window_name: str = "Face Analysis Pipeline",
        log_every_frame: bool = False,
        verbose: bool = False
    ):
        """
        Initialize renderer and logger.
        
        Args:
            jsonl_path: Path to output JSONL log file
            draw_labels: Draw bounding boxes and labels
            show_fps: Display FPS counter
            show_display: Show GUI window (False = headless mode)
            save_video: Save annotated video output
            output_video_path: Path for output video
            width, height: Frame dimensions
            fps: Video frame rate
            window_name: OpenCV window title
            log_every_frame: Log all detections (default: only on attribute change)
            verbose: Detailed logging
        """
        self.jsonl_path = jsonl_path
        self.draw_labels = draw_labels
        self.show_fps = show_fps
        self.show_display = show_display
        self.save_video = save_video
        self.window_name = window_name
        self.log_every_frame = log_every_frame
        self.verbose = verbose
        
        # Video writer
        self.writer = None
        if save_video and output_video_path:
            self._init_video_writer(output_video_path, width, height, fps)
        
        # JSONL log file (buffering=1 for line buffering)
        try:
            self.log_file = open(jsonl_path, "a", buffering=1, encoding='utf-8')
            if verbose:
                print(f"[RendererLogger] Logging to: {jsonl_path}")
        except Exception as e:
            print(f"[RendererLogger] Failed to open log file: {e}", file=sys.stderr)
            self.log_file = None
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_current = 0.0
        self.fps_smoothed = 0.0
        self.fps_alpha = 0.1  # Exponential moving average factor
        
        # Statistics
        self.total_detections_logged = 0
        self.total_frames_rendered = 0
        
        # Track last logged state (for de-duplication)
        self.last_logged_state: dict = {}  # track_id → (age, gender)
        
        # GUI setup
        if self.show_display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            if verbose:
                print(f"[RendererLogger] Display window: {self.window_name}")
    
    def _init_video_writer(self, output_path: str, width: int, height: int, fps: int):
        """Initialize video writer with fallback codecs."""
        # Try multiple codecs (platform-dependent availability)
        codecs = [
            ('mp4v', '.mp4'),  # MPEG-4
            ('avc1', '.mp4'),  # H.264 (better quality)
            ('XVID', '.avi'),  # Xvid
            ('MJPG', '.avi'),  # Motion JPEG (most compatible)
        ]
        
        # Ensure output path has correct extension
        output_path = str(Path(output_path))
        
        for codec, ext in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                
                # Adjust extension if needed
                if not output_path.endswith(ext):
                    base = str(Path(output_path).with_suffix(''))
                    output_path = base + ext
                
                self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if self.writer.isOpened():
                    if self.verbose:
                        print(f"[RendererLogger] Video writer initialized: {codec} → {output_path}")
                    self.output_video_path = output_path
                    return
                else:
                    self.writer = None
            
            except Exception as e:
                if self.verbose:
                    print(f"[RendererLogger] Codec {codec} failed: {e}")
                continue
        
        print(f"[RendererLogger] Failed to initialize video writer", file=sys.stderr)
        self.writer = None
    
    def render(
        self,
        frame,
        annotated_list: List[AnnotatedDetection],
        frame_index: int
    ) -> bool:
        """
        Render annotations on frame and log detections.
        
        Args:
            frame: BGR frame from OpenCV
            annotated_list: List of annotated detections
            frame_index: Current frame number
            
        Returns:
            True to continue, False to stop pipeline
        """
        self.total_frames_rendered += 1
        
        # Update FPS calculation
        self._update_fps()
        
        # Draw annotations on frame
        if self.draw_labels:
            self._draw_annotations(frame, annotated_list)
        
        # Draw FPS and statistics overlay
        if self.show_fps or self.draw_labels:
            self._draw_overlay(frame, annotated_list)
        
        # Log detections to JSONL
        self._log_detections(annotated_list, frame_index)
        
        # Save video frame
        if self.writer and self.writer.isOpened():
            try:
                self.writer.write(frame)
            except Exception as e:
                print(f"[RendererLogger] Video write error: {e}", file=sys.stderr)
        
        # Display frame (if not headless)
        if self.show_display:
            try:
                cv2.imshow(self.window_name, frame)
                
                # Check for quit key (ESC, 'q', 'Q')
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    if self.verbose:
                        print("\n[RendererLogger] User requested quit")
                    return False
            except Exception as e:
                print(f"[RendererLogger] Display error: {e}", file=sys.stderr)
                # Continue without display
        
        return True
    
    def _draw_annotations(self, frame, annotated_list: List[AnnotatedDetection]):
        """Draw bounding boxes and labels on frame."""
        for det in annotated_list:
            # Extract bbox (format: x, y, w, h)
            x, y, w, h = det.bbox
            
            # Draw bounding box
            color = (0, 255, 255)  # Yellow
            thickness = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Build label text
            label_parts = [f"ID {det.track_id}"]
            
            if det.age is not None and det.gender is not None:
                # Both attributes available
                label_parts.append(f"{det.age}y • {det.gender}")
            elif det.age is not None:
                # Only age
                label_parts.append(f"{det.age}y")
            elif det.gender is not None:
                # Only gender
                label_parts.append(det.gender)
            else:
                # No attributes yet
                label_parts.append("analyzing...")
            
            label = ": ".join(label_parts)
            
            # Draw label with background
            label_y = max(20, y - 10)  # Ensure label is visible
            draw_label(
                frame,
                label,
                x, label_y,
                font_scale=0.6,
                thickness=1,
                color=(255, 255, 255),  # White text
                bg_color=(0, 0, 0)       # Black background
            )
    
    def _draw_overlay(self, frame, annotated_list: List[AnnotatedDetection]):
        """Draw FPS and statistics overlay."""
        lines = []
        
        # FPS
        if self.show_fps:
            lines.append(f"FPS: {self.fps_smoothed:.1f}")
        
        # Face count
        lines.append(f"Faces: {len(annotated_list)}")
        
        # Calculate age/gender stats (if available)
        ages = [d.age for d in annotated_list if d.age is not None]
        genders = [d.gender for d in annotated_list if d.gender is not None]
        
        if ages:
            avg_age = sum(ages) / len(ages)
            lines.append(f"Avg Age: {avg_age:.1f}y")
        
        if genders:
            men_count = sum(1 for g in genders if g == 'Man')
            women_count = sum(1 for g in genders if g == 'Woman')
            lines.append(f"M/W: {men_count}/{women_count}")
        
        # Draw semi-transparent background
        overlay = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Calculate overlay dimensions
        line_height = 25
        padding = 10
        max_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] 
                       for line in lines) if lines else 100
        
        overlay_height = len(lines) * line_height + padding * 2
        overlay_width = max_width + padding * 2
        
        # Draw background rectangle
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + overlay_width, 10 + overlay_height),
            (0, 0, 0),
            -1
        )
        
        # Blend with original frame (transparency)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw text lines
        for i, line in enumerate(lines):
            y = 30 + i * line_height
            cv2.putText(
                frame,
                line,
                (20, y),
                font,
                font_scale,
                (0, 255, 0),  # Green text
                thickness,
                cv2.LINE_AA
            )
    
    def _log_detections(self, annotated_list: List[AnnotatedDetection], frame_index: int):
        """Write detections to JSONL log file."""
        if self.log_file is None:
            return
        
        for det in annotated_list:
            # Determine if we should log this detection
            should_log = self.log_every_frame
            
            if not should_log:
                # Only log if attributes changed (de-duplication)
                current_state = (det.age, det.gender)
                last_state = self.last_logged_state.get(det.track_id)
                
                if last_state != current_state:
                    should_log = True
                    self.last_logged_state[det.track_id] = current_state
            
            if should_log:
                try:
                    # Create log entry using factory method
                    log_entry = LogEntry.from_annotated_detection(det, frame_index)
                    
                    # Write as single-line JSON
                    self.log_file.write(log_entry.to_jsonl() + "\n")
                    self.total_detections_logged += 1
                
                except Exception as e:
                    print(f"[RendererLogger] Log write error: {e}", file=sys.stderr)
    
    def _update_fps(self):
        """Update FPS calculation with exponential moving average."""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        
        if elapsed > 0:
            # Instantaneous FPS
            self.fps_current = 1.0 / elapsed
            
            # Smoothed FPS (exponential moving average)
            self.fps_smoothed = (
                self.fps_alpha * self.fps_current +
                (1 - self.fps_alpha) * self.fps_smoothed
            )
        
        self.start_time = time.time()
    
    def cleanup_logs(self, active_track_ids: List[int]):
        """Remove logged state for inactive tracks (prevent memory leak)."""
        stale_ids = [tid for tid in self.last_logged_state.keys() 
                     if tid not in active_track_ids]
        
        for tid in stale_ids:
            del self.last_logged_state[tid]
    
    def close(self):
        """Clean up resources."""
        if self.verbose:
            print("\n[RendererLogger] Shutting down...")
        
        # Close log file
        if self.log_file:
            try:
                self.log_file.close()
                if self.verbose:
                    print(f"[RendererLogger] Closed log file: {self.jsonl_path}")
            except:
                pass
        
        # Release video writer
        if self.writer:
            try:
                self.writer.release()
                if self.verbose:
                    print(f"[RendererLogger] Saved video: {self.output_video_path}")
            except:
                pass
        
        # Close OpenCV windows
        if self.show_display:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        # Print statistics
        if self.verbose:
            print(f"[RendererLogger] Rendered {self.total_frames_rendered} frames")
            print(f"[RendererLogger] Logged {self.total_detections_logged} detections")
    
    def get_stats(self) -> dict:
        """Get rendering statistics."""
        return {
            'total_frames_rendered': self.total_frames_rendered,
            'total_detections_logged': self.total_detections_logged,
            'fps_current': self.fps_current,
            'fps_smoothed': self.fps_smoothed,
            'active_tracks_logged': len(self.last_logged_state),
        }
    
    def print_stats(self):
        """Print rendering statistics."""
        stats = self.get_stats()
        print("\n=== Renderer/Logger Statistics ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key:30s}: {value:.2f}")
            else:
                print(f"  {key:30s}: {value}")
        print("=" * 40 + "\n")


# ============================================================
# Utility Functions
# ============================================================

def test_renderer():
    """Test renderer with sample data."""
    import numpy as np
    from .data_models import AnnotatedDetection
    
    print("=== Testing RendererLogger ===\n")
    
    # Create renderer
    renderer = RendererLogger(
        jsonl_path="test_logs.jsonl",
        draw_labels=True,
        show_fps=True,
        show_display=True,
        save_video=False,
        verbose=True
    )
    
    # Create fake frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (400, 400), (100, 100, 255), -1)
    
    # Create fake detections
    detections = [
        AnnotatedDetection(
            bbox=(200, 150, 200, 250),
            confidence=0.9,
            track_id=1,
            age=34,
            gender="Man"
        ),
        AnnotatedDetection(
            bbox=(420, 180, 180, 220),
            confidence=0.85,
            track_id=2,
            age=28,
            gender="Woman"
        )
    ]
    
    # Render frames
    print("Rendering 100 frames (press Q to quit)...")
    for i in range(100):
        should_continue = renderer.render(frame, detections, frame_index=i)
        
        if not should_continue:
            print("User quit")
            break
        
        time.sleep(0.033)  # ~30 FPS
    
    # Cleanup
    renderer.close()
    renderer.print_stats()
    
    print("\nTest complete")
    print(f"Check test_logs.jsonl for output")


if __name__ == "__main__":
    test_renderer()
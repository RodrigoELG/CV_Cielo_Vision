# src/frame_grabber.py
"""
Background frame capture thread for continuous video acquisition.

Uses queue.Queue for thread-safe frame passing between:
- Producer (this thread): Captures frames continuously
- Consumer (main loop): Processes frames at its own pace

Technical Notes:
- Queue prevents race conditions (deque is NOT thread-safe for concurrent read/write)
- FramePacket includes timestamp to detect stale frames
- Auto-reconnection with exponential backoff for RTSP streams
- Platform-agnostic backend selection (CAP_V4L2 → CAP_ANY fallback)
- Frame skipping prevents queue overflow (drops old frames, keeps latest)

Performance:
- Queue maxsize=2 balances latency (fresh frames) vs dropped frames
- Producer runs at camera FPS (e.g., 30fps)
- Consumer processes at pipeline speed (e.g., 20fps)
- Difference handled by queue drop logic
"""

import cv2
import threading
import time
import queue
import sys
from typing import Optional, Union
from pathlib import Path

from .data_models import FramePacket


class FrameGrabber(threading.Thread):
    """
    Background thread for continuous frame capture.
    
    Supports:
    - Webcams (index 0, 1, 2, ...)
    - Video files (.mp4, .avi, ...)
    - RTSP streams (rtsp://...)
    - HTTP streams (http://...)
    """
    
    def __init__(
        self,
        source: Union[str, int],
        width: int,
        height: int,
        queue_size: int = 2,
        backend: Optional[str] = None,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
        verbose: bool = False
    ):
        """
        Initialize frame grabber.
        
        Args:
            source: Camera index (0, 1, ...), file path, or stream URL
            width: Target frame width in pixels
            height: Target frame height in pixels
            queue_size: Max frames in queue (2 = keep 2 most recent)
            backend: OpenCV backend ('V4L2'|'DSHOW'|'ANY'|None for auto)
            reconnect_attempts: Max reconnection tries for streams
            reconnect_delay: Initial delay between reconnections (seconds)
            verbose: Enable detailed logging
        """
        super().__init__(daemon=True, name="FrameGrabber")
        
        # Configuration
        self.source = self._parse_source(source)
        self.width = width
        self.height = height
        self.backend = self._select_backend(backend)
        self.max_reconnect_attempts = reconnect_attempts
        self.reconnect_delay_base = reconnect_delay
        self.verbose = verbose
        
        # State
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.frame_idx = 0
        self.reconnect_count = 0
        self.total_frames_captured = 0
        self.total_frames_dropped = 0
        
        # Statistics
        self.fps_actual = 0.0
        self._fps_start_time = time.time()
        self._fps_frame_count = 0
        
        # Error tracking
        self.last_error = None
        self.error_count = 0
    
    def _parse_source(self, source: Union[str, int]) -> Union[str, int]:
        """Parse and validate source parameter."""
        if isinstance(source, int):
            return source
        
        # Try to convert string to int (camera index)
        if isinstance(source, str) and source.isdigit():
            return int(source)
        
        # File or stream URL
        if isinstance(source, str):
            # Validate file exists if it's a local path
            if not source.startswith(('rtsp://', 'http://', 'https://')):
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(f"Video file not found: {source}")
            return source
        
        raise ValueError(f"Invalid source type: {type(source)}")
    
    def _select_backend(self, backend: Optional[str]) -> int:
        """
        Select OpenCV video capture backend.
        
        Technical Note:
        - CAP_V4L2: Linux cameras (best performance)
        - CAP_DSHOW: Windows cameras
        - CAP_ANY: Universal fallback
        """
        if backend is None:
            # Auto-detect based on platform and source type
            if isinstance(self.source, int):
                # Camera - use platform-specific backend
                if sys.platform.startswith('linux'):
                    return cv2.CAP_V4L2
                elif sys.platform.startswith('win'):
                    return cv2.CAP_DSHOW
            # File/stream - use any backend
            return cv2.CAP_ANY
        
        # Manual backend selection
        backend_map = {
            'V4L2': cv2.CAP_V4L2,
            'DSHOW': cv2.CAP_DSHOW,
            'ANY': cv2.CAP_ANY,
            'FFMPEG': cv2.CAP_FFMPEG,
            'GSTREAMER': cv2.CAP_GSTREAMER,
        }
        
        backend_upper = backend.upper()
        if backend_upper not in backend_map:
            print(f"Unknown backend '{backend}', using CAP_ANY", file=sys.stderr)
            return cv2.CAP_ANY
        
        return backend_map[backend_upper]
    
    def run(self):
        """Main capture loop (runs in background thread)."""
        if self.verbose:
            print(f"[FrameGrabber] Thread started for source: {self.source}")
        
        # Open capture device
        if not self._open_capture():
            print(f"[FrameGrabber] Failed to open source: {self.source}", file=sys.stderr)
            return
        
        self.running = True
        
        # Capture loop
        while self.running:
            # Check if capture is still valid
            if self.cap is None or not self.cap.isOpened():
                if not self._reconnect():
                    print(f"[FrameGrabber] Exhausted reconnection attempts", file=sys.stderr)
                    break
                continue
            
            # Read frame
            ok, frame = self.cap.read()
            
            if not ok:
                self.error_count += 1
                self.last_error = "Failed to read frame"
                
                if self.verbose:
                    print(f"[FrameGrabber] Frame read failed (error #{self.error_count})")
                
                # Attempt reconnection for streams
                if self._is_stream():
                    if not self._reconnect():
                        break
                else:
                    # For files, end of video is expected
                    if self.verbose:
                        print(f"[FrameGrabber] End of video file reached")
                    break
                
                continue
            
            # Reset error counter on successful read
            if self.error_count > 0:
                self.error_count = 0
                self.reconnect_count = 0
            
            # Update statistics
            self.frame_idx += 1
            self.total_frames_captured += 1
            self._update_fps()
            
            # Create frame packet with metadata
            packet = FramePacket(
                frame=frame,
                frame_idx=self.frame_idx,
                timestamp=time.time()
            )
            
            # Put frame in queue (non-blocking, drop old frames if full)
            try:
                self.frame_queue.put_nowait(packet)
            except queue.Full:
                # Queue full - drop oldest frame and try again
                try:
                    self.frame_queue.get_nowait()  # Drop oldest
                    self.frame_queue.put_nowait(packet)  # Add new
                    self.total_frames_dropped += 1
                    
                    if self.verbose and self.total_frames_dropped % 10 == 0:
                        print(f"[FrameGrabber] Dropped {self.total_frames_dropped} frames (queue overflow)")
                except:
                    pass
        
        # Cleanup
        self._close_capture()
        if self.verbose:
            print(f"[FrameGrabber] Thread stopped. Captured {self.total_frames_captured} frames, dropped {self.total_frames_dropped}")
    
    def _open_capture(self) -> bool:
        """Open video capture device."""
        try:
            self.cap = cv2.VideoCapture(self.source, self.backend)
            
            if not self.cap.isOpened():
                return False
            
            # Set resolution (may not work for all sources)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Set buffer size to 1 to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get actual resolution (may differ from requested)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            if self.verbose:
                print(f"[FrameGrabber] Opened: {self.source}")
                print(f"[FrameGrabber] Resolution: {actual_width}×{actual_height} (requested: {self.width}×{self.height})")
                if actual_fps > 0:
                    print(f"[FrameGrabber] FPS: {actual_fps:.1f}")
            
            return True
            
        except Exception as e:
            self.last_error = str(e)
            print(f"[FrameGrabber] Error opening capture: {e}", file=sys.stderr)
            return False
    
    def _close_capture(self):
        """Close video capture and release resources."""
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            finally:
                self.cap = None
    
    def _is_stream(self) -> bool:
        """Check if source is a network stream (vs file or camera)."""
        if isinstance(self.source, str):
            return self.source.startswith(('rtsp://', 'http://', 'https://'))
        return False
    
    def _reconnect(self) -> bool:
        """
        Attempt to reconnect to source with exponential backoff.
        
        Technical Note:
        Exponential backoff: 2s → 4s → 8s → 16s → 30s (capped)
        Prevents hammering failing RTSP servers.
        """
        if self.reconnect_count >= self.max_reconnect_attempts:
            print(f"[FrameGrabber] Max reconnection attempts ({self.max_reconnect_attempts}) reached", file=sys.stderr)
            return False
        
        self.reconnect_count += 1
        
        # Calculate delay with exponential backoff
        delay = min(self.reconnect_delay_base * (2 ** (self.reconnect_count - 1)), 30.0)
        
        print(f"[FrameGrabber] Reconnecting (attempt {self.reconnect_count}/{self.max_reconnect_attempts})...")
        
        # Close existing capture
        self._close_capture()
        
        # Wait before reconnecting
        time.sleep(delay)
        
        # Try to reopen
        if self._open_capture():
            print(f"[FrameGrabber] Reconnection successful")
            return True
        else:
            print(f"[FrameGrabber] Reconnection failed", file=sys.stderr)
            return False
    
    def _update_fps(self):
        """Update FPS calculation."""
        self._fps_frame_count += 1
        elapsed = time.time() - self._fps_start_time
        
        # Update FPS every second
        if elapsed >= 1.0:
            self.fps_actual = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_start_time = time.time()
    
    def get_frame(self, timeout: float = 0.1) -> Optional[FramePacket]:
        """
        Get latest frame from queue (thread-safe).
        
        Args:
            timeout: Max time to wait for frame (seconds)
            
        Returns:
            FramePacket with frame and metadata, or None if timeout/empty
            
        Technical Note:
        Uses blocking Queue.get() with timeout instead of deque
        to prevent race conditions between producer/consumer threads.
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Signal thread to stop and wait for cleanup."""
        if self.verbose:
            print("[FrameGrabber] Stopping...")
        
        self.running = False
        
        # Wait for thread to finish (with timeout)
        self.join(timeout=2.0)
        
        if self.is_alive():
            print("[FrameGrabber] Thread did not stop cleanly", file=sys.stderr)
    
    def get_stats(self) -> dict:
        """Get capture statistics."""
        return {
            'source': self.source,
            'running': self.running,
            'frames_captured': self.total_frames_captured,
            'frames_dropped': self.total_frames_dropped,
            'current_frame_idx': self.frame_idx,
            'fps_actual': self.fps_actual,
            'queue_size': self.frame_queue.qsize(),
            'reconnect_count': self.reconnect_count,
            'error_count': self.error_count,
            'last_error': self.last_error,
        }
    
    def print_stats(self):
        """Print capture statistics."""
        stats = self.get_stats()
        print("\n=== Frame Grabber Statistics ===")
        for key, value in stats.items():
            print(f"  {key:20s}: {value}")
        print("=" * 40 + "\n")


# ============================================================
# Utility Functions
# ============================================================

def test_camera(camera_index: int = 0, duration: float = 5.0):
    """
    Test camera capture for a few seconds.
    
    Args:
        camera_index: Camera index to test
        duration: How long to test (seconds)
    """
    print(f"Testing camera {camera_index} for {duration} seconds...")
    
    grabber = FrameGrabber(
        source=camera_index,
        width=640,
        height=480,
        verbose=True
    )
    
    grabber.start()
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            packet = grabber.get_frame(timeout=1.0)
            
            if packet is None:
                print("No frame received")
                continue
            
            frame_count += 1
            print(f"Frame {frame_count}: {packet.frame.shape}, age={packet.age_ms:.1f}ms")
            
            time.sleep(0.1)  # Don't consume all frames
    
    finally:
        grabber.stop()
        grabber.print_stats()


def test_rtsp_stream(url: str, duration: float = 10.0):
    """
    Test RTSP stream capture.
    
    Args:
        url: RTSP stream URL
        duration: How long to test (seconds)
    """
    print(f"Testing RTSP stream: {url}")
    
    grabber = FrameGrabber(
        source=url,
        width=1280,
        height=720,
        reconnect_attempts=3,
        verbose=True
    )
    
    grabber.start()
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            packet = grabber.get_frame(timeout=2.0)
            
            if packet is None:
                print("No frame received (timeout)")
                continue
            
            frame_count += 1
            
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"Captured {frame_count} frames, FPS: {grabber.fps_actual:.1f}")
    
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    finally:
        grabber.stop()
        grabber.print_stats()


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    import sys
    
    print("=== Frame Grabber Test ===\n")
    
    # Test camera
    if len(sys.argv) > 1 and sys.argv[1] == 'rtsp':
        # Test RTSP: python frame_grabber.py rtsp rtsp://camera.local/stream
        if len(sys.argv) > 2:
            test_rtsp_stream(sys.argv[2], duration=10.0)
        else:
            print("Usage: python frame_grabber.py rtsp <rtsp_url>")
    else:
        # Test camera: python frame_grabber.py [camera_index]
        camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        test_camera(camera_idx, duration=5.0)
# src/main.py
"""
Face Analysis Pipeline - Main Entry Point

Orchestrates the complete pipeline:
1. Configuration loading
2. Frame capture (background thread)
3. Face detection (YOLO)
4. Face tracking (SORT/ByteTrack)
5. Attribute analysis (DeepFace with caching)
6. Rendering & logging (JSONL + video)

Technical Notes:
- Multi-threaded: FrameGrabber runs in background, main loop processes
- Modular: Each component is a separate, testable module
- Configurable: All parameters via config.py (CLI + YAML)
- Resilient: Comprehensive error handling, graceful shutdown
- Observable: Statistics tracking, performance monitoring

Architecture:
    Thread A: FrameGrabber → Queue → Thread B: Main Loop
    Main Loop: detect → track → analyze → render → log
"""

import os
import sys
import signal
import time
from pathlib import Path

# Suppress TensorFlow/DeepFace warnings BEFORE imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TQDM_DISABLE"] = "1"

# Import our modules
from .config import load_config
from .frame_grabber import FrameGrabber
from .detector_tracker import FaceDetectorTracker
from .attributes import AttributeAnalyzer
from .renderer_logger import RendererLogger
from .utils import Timer, ensure_dir


class FacePipeline:
    """
    Main pipeline orchestrator.
    
    Manages lifecycle of all components and coordinates data flow.
    """
    
    def __init__(self, config):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Config object from config.py
        """
        self.config = config
        self.running = False
        self.components_initialized = False
        
        # Performance tracking
        self.total_frames_processed = 0
        self.start_time = time.time()
        
        print("\n" + "="*70)
        print("FACE ANALYSIS PIPELINE")
        print("="*70)
        
        # Initialize components
        self._init_components()
        
        print("="*70 + "\n")
    
    def _init_components(self):
        """Initialize all pipeline components."""
        try:
            # 1. Frame Grabber (background thread)
            print("\n[1/4] Initializing Frame Grabber...")
            self.frame_grabber = FrameGrabber(
                source=self.config.source,
                width=self.config.width,
                height=self.config.height,
                queue_size=self.config.frame_queue_size,
                verbose=self.config.verbose
            )
            print("  Frame grabber ready")
            
            # 2. Detector + Tracker (YOLO + SORT)
            print("\n[2/4] Initializing Detector & Tracker...")
            self.detector = FaceDetectorTracker(
                model_path=self.config.yolo_model,
                device=self.config.device,
                min_confidence=self.config.min_confidence,
                iou_threshold=self.config.iou_threshold,
                max_age=self.config.tracker_max_age,
                min_hits=self.config.tracker_min_hits
            )
            print("  Detector & tracker ready")
            
            # 3. Attribute Analyzer (DeepFace with caching)
            print("\n[3/4] Initializing Attribute Analyzer...")
            self.analyzer = AttributeAnalyzer(
                analyze_interval_frames=self.config.analyze_interval_frames,
                min_crop_size=self.config.min_crop_size,
                crop_padding=self.config.crop_padding,
                backend=self.config.deepface_backend,
                enforce_detection=self.config.deepface_detector_enforce,
                debug=self.config.debug,
                save_debug_crops=self.config.save_debug_crops,
                debug_crops_dir=self.config.debug_crops_dir,
                verbose=self.config.verbose
            )
            print("  Attribute analyzer ready")
            
            # 4. Renderer + Logger (visualization + JSONL)
            print("\n[4/4] Initializing Renderer & Logger...")
            self.renderer = RendererLogger(
                jsonl_path=self.config.jsonl_path,
                draw_labels=self.config.draw_labels,
                show_fps=self.config.show_fps,
                show_display=self.config.show_display,
                save_video=self.config.save_video,
                output_video_path=self.config.output_video_path if self.config.save_video else None,
                width=self.config.width,
                height=self.config.height,
                fps=self.config.fps_target,
                verbose=self.config.verbose
            )
            print("  Renderer & logger ready")
            
            self.components_initialized = True
            
        except Exception as e:
            print(f"\nFailed to initialize pipeline: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def run(self):
        """
        Main processing loop.
        
        Flow:
        1. Start frame grabber thread
        2. Get frame from queue
        3. Detect faces (YOLO)
        4. Track faces (SORT)
        5. Analyze attributes (DeepFace)
        6. Render & log
        7. Repeat
        """
        if not self.components_initialized:
            print("Components not initialized", file=sys.stderr)
            return
        
        # Start frame grabber in background
        print("Starting frame capture thread...")
        self.frame_grabber.start()
        time.sleep(0.5)  # Give grabber time to start
        
        self.running = True
        self.start_time = time.time()
        
        print("\n" + "="*70)
        print("PIPELINE RUNNING")
        print("="*70)
        print("Controls:")
        print("  • Press 'Q' or ESC to quit")
        print("  • Check logs at:", self.config.jsonl_path)
        if self.config.save_video:
            print("  • Video saving to:", self.config.output_video_path)
        print("="*70 + "\n")
        
        # Counters for periodic operations
        frame_count = 0
        no_frame_count = 0
        last_cleanup_frame = 0
        last_stats_frame = 0
        
        # Performance tracking
        frame_timer = Timer()
        
        try:
            while self.running:
                frame_timer.reset()
                
                # Get frame from grabber thread
                packet = self.frame_grabber.get_frame(timeout=0.1)
                
                if packet is None:
                    no_frame_count += 1
                    
                    # Exit if too many consecutive failures
                    if no_frame_count >= 100:
                        print("\nNo frames received for 10 seconds, exiting...", file=sys.stderr)
                        break
                    
                    continue
                
                # Reset no-frame counter on successful read
                no_frame_count = 0
                
                # Extract frame data
                frame = packet.frame
                frame_idx = packet.frame_idx
                
                # Check if frame is stale (queue backup)
                if packet.is_stale(threshold_ms=100):
                    if self.config.debug:
                        print(f"Stale frame detected: {packet.age_ms:.1f}ms old")
                
                # Process frame
                try:
                    # Step 1: Detect and track faces
                    tracked_detections = self.detector.detect_and_track(frame)
                    
                    # Step 2: Analyze attributes (with caching)
                    annotated_detections = self.analyzer.analyze(
                        tracked_detections,
                        frame,
                        frame_idx
                    )
                    
                    # Step 3: Render and log
                    should_continue = self.renderer.render(
                        frame,
                        annotated_detections,
                        frame_idx
                    )
                    
                    if not should_continue:
                        print("\n[Pipeline] User requested quit")
                        break
                
                except Exception as e:
                    print(f"\nError processing frame {frame_idx}: {e}", file=sys.stderr)
                    if self.config.debug:
                        import traceback
                        traceback.print_exc()
                    continue
                
                frame_count += 1
                self.total_frames_processed += 1
                
                # Periodic cache cleanup (every ~10 seconds @ 30fps)
                if frame_idx - last_cleanup_frame >= 300:
                    active_ids = [d.track_id for d in tracked_detections]
                    self.analyzer.cleanup_old_tracks(active_ids)
                    self.renderer.cleanup_logs(active_ids)
                    last_cleanup_frame = frame_idx
                
                # Periodic statistics (every 30 seconds)
                if self.config.verbose and frame_idx - last_stats_frame >= 900:
                    self._print_runtime_stats()
                    last_stats_frame = frame_idx
                
                # Warn on slow frames (> 50ms = < 20 FPS)
                frame_time = frame_timer.elapsed_ms()
                if frame_time > 50 and self.config.debug:
                    print(f"Slow frame {frame_idx}: {frame_time:.1f}ms")
        
        except KeyboardInterrupt:
            print("\n\n[Pipeline] Interrupted by user (Ctrl+C)")
        
        except Exception as e:
            print(f"\n\nPipeline error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of all components."""
        print("\n" + "="*70)
        print("SHUTTING DOWN")
        print("="*70)
        
        self.running = False
        
        # Stop frame grabber
        if hasattr(self, 'frame_grabber'):
            print("\n[1/4] Stopping frame grabber...")
            self.frame_grabber.stop()
            print(" Frame grabber stopped")
        
        # Close renderer (saves video, closes logs)
        if hasattr(self, 'renderer'):
            print("\n[2/4] Closing renderer & logger...")
            self.renderer.close()
            print(" Renderer & logger closed")
        
        # Print final statistics
        print("\n[3/4] Generating final statistics...")
        self._print_final_stats()
        print(" Statistics generated")
        
        print("\n[4/4] Cleanup complete")
        print("\n" + "="*70)
        print("PIPELINE STOPPED")
        print("="*70 + "\n")
    
    def _print_runtime_stats(self):
        """Print runtime statistics."""
        print("\n" + "="*70)
        print("RUNTIME STATISTICS")
        print("="*70)
        
        # Frame grabber stats
        if hasattr(self, 'frame_grabber'):
            grabber_stats = self.frame_grabber.get_stats()
            print("\nFrame Grabber:")
            print(f"  Frames captured  : {grabber_stats['frames_captured']}")
            print(f"  Frames dropped   : {grabber_stats['frames_dropped']}")
            print(f"  Actual FPS       : {grabber_stats['fps_actual']:.1f}")
        
        # Analyzer stats
        if hasattr(self, 'analyzer'):
            analyzer_stats = self.analyzer.get_stats()
            print("\nAttribute Analyzer:")
            print(f"  Total analyses   : {analyzer_stats['total_analyses']}")
            print(f"  Cache hit rate   : {analyzer_stats['cache_hit_rate']}")
            print(f"  Failed analyses  : {analyzer_stats['failed_analyses']}")
        
        # Renderer stats
        if hasattr(self, 'renderer'):
            renderer_stats = self.renderer.get_stats()
            print("\nRenderer/Logger:")
            print(f"  Frames rendered  : {renderer_stats['total_frames_rendered']}")
            print(f"  Detections logged: {renderer_stats['total_detections_logged']}")
        
        print("="*70 + "\n")
    
    def _print_final_stats(self):
        """Print final statistics at shutdown."""
        elapsed = time.time() - self.start_time
        avg_fps = self.total_frames_processed / elapsed if elapsed > 0 else 0.0
        
        print("\n📊 Final Statistics:")
        print(f"  Total runtime       : {elapsed:.1f}s")
        print(f"  Frames processed    : {self.total_frames_processed}")
        print(f"  Average FPS         : {avg_fps:.1f}")
        
        # Component stats
        if hasattr(self, 'frame_grabber'):
            grabber_stats = self.frame_grabber.get_stats()
            print(f"  Frames dropped      : {grabber_stats['frames_dropped']}")
        
        if hasattr(self, 'analyzer'):
            analyzer_stats = self.analyzer.get_stats()
            print(f"  Attribute analyses  : {analyzer_stats['total_analyses']}")
            print(f"  Cache hit rate      : {analyzer_stats['cache_hit_rate']}")
        
        if hasattr(self, 'renderer'):
            renderer_stats = self.renderer.get_stats()
            print(f"  Detections logged   : {renderer_stats['total_detections_logged']}")
        
        print(f"\nOutput files:")
        print(f"  Logs: {self.config.jsonl_path}")
        if self.config.save_video:
            print(f"  Video: {self.config.output_video_path}")
        if self.config.save_debug_crops:
            print(f"  Debug crops: {self.config.debug_crops_dir}")


def main():
    """
    Main entry point.
    
    Flow:
    1. Load configuration (CLI + YAML)
    2. Print configuration
    3. Validate environment
    4. Create pipeline
    5. Setup signal handlers
    6. Run pipeline
    """
    
    # ASCII banner
    print("\n" + "="*70)
    print("""
    ███████╗ █████╗  ██████╗███████╗    ██████╗ ██╗██████╗ ███████╗██╗     ██╗███╗   ██╗███████╗
    ██╔════╝██╔══██╗██╔════╝██╔════╝    ██╔══██╗██║██╔══██╗██╔════╝██║     ██║████╗  ██║██╔════╝
    █████╗  ███████║██║     █████╗      ██████╔╝██║██████╔╝█████╗  ██║     ██║██╔██╗ ██║█████╗  
    ██╔══╝  ██╔══██║██║     ██╔══╝      ██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝  
    ██║     ██║  ██║╚██████╗███████╗    ██║     ██║██║     ███████╗███████╗██║██║ ╚████║███████╗
    ╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝    ╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝
    """)
    print("="*70 + "\n")
    
    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print configuration (if verbose)
    if config.verbose or config.debug:
        config.print_config()
    
    # Validate YOLO model exists
    if not Path(config.yolo_model).exists():
        print(f"YOLO model not found: {config.yolo_model}", file=sys.stderr)
        print("   Please download yolov8n-face-lindevs.pt to weights/ directory", file=sys.stderr)
        sys.exit(1)
    
    # Create pipeline
    try:
        pipeline = FacePipeline(config)
    except Exception as e:
        print(f"Failed to create pipeline: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        """Handle interrupt signals (Ctrl+C, SIGTERM)."""
        print(f"\n\n  Received signal {sig}, shutting down gracefully...")
        pipeline.running = False
    
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Kill command
    
    # Run pipeline
    try:
        pipeline.run()
    except Exception as e:
        print(f"\nPipeline crashed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Exit cleanly
    sys.exit(0)


# For direct execution
if __name__ == "__main__":
    main()
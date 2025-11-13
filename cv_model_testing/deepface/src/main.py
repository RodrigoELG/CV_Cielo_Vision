# src/main.py
import threading
import queue
import time
import signal
from .config import load_config
from .frame_grabber import FrameGrabber
from .detector_tracker import DetectorTracker
from .attributes import AttributeAnalyzer
from .renderer_logger import RendererLogger
from .data_models import AnnotatedDetection

def main():
    cfg = load_config()
    device = cfg.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Queues / shared buffers
    latest_frame_queue = queue.Queue(maxsize=1)
    tracked_queue = queue.Queue(maxsize=2)
    annotated_queue = queue.Queue(maxsize=2)

    # Frame grabber
    grabber = FrameGrabber(cfg.source, cfg.width, cfg.height)
    grabber.start()

    # Detector + tracker
    from sort import Sort  # or your own SORT implementation
    tracker_constructor = Sort  # alias
    detector_tracker = DetectorTracker(weights_path="weights/yolov8n-face.pt",
                                       device=device,
                                       min_confidence=cfg.min_confidence,
                                       tracker_constructor=tracker_constructor)

    attribute_analyzer = AttributeAnalyzer(analyze_interval_frames=cfg.analyze_interval_frames,
                                           age_model=None,  # load age_model separately if needed
                                           gender_model=None)

    renderer = RendererLogger(jsonl_path=cfg.jsonl_path,
                              draw_labels=cfg.draw_labels,
                              save_video=cfg.save_video,
                              output_video_path=cfg.output_video_path,
                              width=cfg.width,
                              height=cfg.height,
                              fps=30)

    stop_event = threading.Event()

    def detector_tracker_loop():
        frame_index = 0
        while not stop_event.is_set():
            frame = grabber.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            tracked_list = detector_tracker.process(frame)
            try:
                tracked_queue.put((frame, frame_index, tracked_list), timeout=0.1)
            except queue.Full:
                pass
            frame_index += 1

    def analyzer_loop():
        while not stop_event.is_set():
            try:
                frame, frame_index, tracked_list = tracked_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            annotated_list = attribute_analyzer.analyze(tracked_list, frame, frame_index)
            try:
                annotated_queue.put((frame, frame_index, annotated_list), timeout=0.1)
            except queue.Full:
                pass

    t1 = threading.Thread(target=detector_tracker_loop, daemon=True)
    t2 = threading.Thread(target=analyzer_loop, daemon=True)
    t1.start()
    t2.start()

    def signal_handler(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    frame_idx = 0
    while not stop_event.is_set():
        try:
            frame, frame_idx, annotated_list = annotated_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        continue_loop = renderer.render(frame, annotated_list, frame_idx)
        if not continue_loop:
            stop_event.set()
        frame_idx += 1

    # Shutdown
    grabber.stop()
    t1.join(timeout=1)
    t2.join(timeout=1)
    renderer.close()

if __name__ == "__main__":
    main()

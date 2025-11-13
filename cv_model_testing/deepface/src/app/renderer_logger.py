# src/renderer_logger.py
import cv2
import json
import datetime
from .utils import draw_label, FPSCounter
from .data_models import LogEntry

class RendererLogger:
    def __init__(self, jsonl_path, draw_labels=True, save_video=False, output_video_path=None, width=1280, height=720, fps=30):
        self.jsonl_path = jsonl_path
        self.draw_labels = draw_labels
        self.save_video = save_video
        self.output_video_path = output_video_path
        self.fps_counter = FPSCounter()
        if save_video and output_video_path:
            self.writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        else:
            self.writer = None
        self.log_file = open(jsonl_path, "a", buffering=1)

    def render(self, frame, annotated_list, frame_index):
        self.fps_counter.tick()
        for ad in annotated_list:
            x0, y0, x1, y1 = ad.bbox
            label = f"ID{ad.track_id}"
            if ad.age is not None and ad.gender is not None:
                label += f": {int(ad.age)}y / {ad.gender}"
            if self.draw_labels:
                draw_label(frame, label, x0, y0-10)
            # log entry
            entry = LogEntry(
                track_id=ad.track_id,
                age=ad.age,
                gender=ad.gender,
                timestamp=datetime.datetime.utcnow(),
                frame_index=frame_index,
                bbox=ad.bbox,
                confidence=ad.confidence
            )
            self.log_file.write(entry.json() + "\n")

        if self.writer:
            self.writer.write(frame)

        cv2.imshow("Face-Age/Gender Tracker", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            return False
        return True

    def close(self):
        if self.writer:
            self.writer.release()
        self.log_file.close()
        cv2.destroyAllWindows()

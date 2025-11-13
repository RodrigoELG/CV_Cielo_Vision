# src/detector_tracker.py
import numpy as np
from ultralytics import YOLO
# import SORT (you will put your implementation or pip install)
from .data_models import TrackedDetection

class DetectorTracker:
    def __init__(self, weights_path, device, min_confidence, tracker_constructor):
        self.model = YOLO(weights_path, device=device)
        self.min_conf = min_confidence
        self.tracker = tracker_constructor()  # instantiate SORT

    def process(self, frame):
        results = self.model.predict(frame, conf=self.min_conf, verbose=False)
        tracked_list = []
        if results and hasattr(results[0], "boxes"):
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy().astype(float)
            dets = []
            for (x0, y0, x1, y1), c in zip(boxes, confs):
                if c < self.min_conf:
                    continue
                dets.append([x0, y0, x1, y1, c])
            # convert to NumPy for tracker
            dets_arr = np.array(dets)
            tracks = self.tracker.update(dets_arr)  # expects Nx5 → [[x0,y0,x1,y1,trackID],...]
            for trk in tracks:
                x0, y0, x1, y1, tid = trk.astype(int).tolist()
                # find confidence by matching dets? Simplify: use first match
                c = next((d[4] for d in dets if d[0]==x0 and d[1]==y0 and d[2]==x1 and d[3]==y1), 0.0)
                tracked_list.append(TrackedDetection(bbox=(x0,y0,x1,y1), confidence=float(c), track_id=int(tid)))
        return tracked_list

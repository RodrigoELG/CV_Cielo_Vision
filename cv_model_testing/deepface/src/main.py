# src/main.py
"""Real-time webcam face detection + tracking (MediaPipe + Kalman Filter MVP)."""
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU before Mediapipe import

import cv2
import mediapipe as mp
import time
import json
import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter

# Path for JSON log file
LOG_PATH = "/home/rodrigo/Documents/CV_Cielo_Vision/cv_model_testing/deepface/logs/events.jsonl"

# Overwrite (clear) the previous log on every new run
if os.path.exists(LOG_PATH):
    open(LOG_PATH, "w").close()

# Keep track of which IDs we’ve already logged
logged_ids = set()

# -------------- Helper Functions --------------

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter / union if union > 0 else 0.0

# -------------- Tracking Class --------------

class Track:
    _id_counter = 0
    def __init__(self, bbox):
        self.id = Track._id_counter
        Track._id_counter += 1

        # Kalman Filter state = [cx, cy, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1.,0.,1.,0.],
                              [0.,1.,0.,1.],
                              [0.,0.,1.,0.],
                              [0.,0.,0.,1.]])
        self.kf.H = np.array([[1.,0.,0.,0.],
                              [0.,1.,0.,0.]])
        self.kf.P *= 1000.
        self.kf.R *= 10.

        cx = bbox[0] + bbox[2]/2.0
        cy = bbox[1] + bbox[3]/2.0
        self.kf.x = np.array([[cx],[cy],[0.],[0.]])
        self.bbox = bbox
        self.lost = 0

    def update(self, bbox=None):
        if bbox is not None:
            cx = bbox[0] + bbox[2]/2.0
            cy = bbox[1] + bbox[3]/2.0
            self.kf.predict()
            self.kf.update(np.array([[cx],[cy]]))
            self.bbox = bbox
            self.lost = 0
        else:
            self.kf.predict()
            self.lost += 1

# -------------- Detection --------------

def detect_faces(frame, face_detector):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)
    boxes = []
    if results.detections:
        ih, iw, _ = frame.shape
        for det in results.detections:
            bbox_rel = det.location_data.relative_bounding_box
            x = int(bbox_rel.xmin * iw)
            y = int(bbox_rel.ymin * ih)
            w = int(bbox_rel.width * iw)
            h = int(bbox_rel.height * ih)
            boxes.append([x, y, w, h])
    return boxes

# -------------- Main Loop --------------

def main():
    mp_fd = mp.solutions.face_detection
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    with mp_fd.FaceDetection(model_selection=0,
                             min_detection_confidence=0.5) as detector:

        tracks = []
        frame_id = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Frame not captured (ret=False or frame=None)")
                continue

            # Optional: print frame shape to confirm resolution
            print(f"[DEBUG] Frame shape: {frame.shape}")

            frame_id += 1

            DETECT_INTERVAL = 10
            if frame_id % DETECT_INTERVAL == 0:
                boxes = detect_faces(frame, detector)
                assigned = []
                for box in boxes:
                    best_iou = 0.0
                    best_t = None
                    for t in tracks:
                        score = iou(box, t.bbox)
                        if score > best_iou:
                            best_iou = score
                            best_t = t
                    if best_iou > 0.3 and best_t is not None:
                        best_t.update(box)
                        assigned.append(best_t)
                    else:
                        new_t = Track(box)
                        tracks.append(new_t)
                        assigned.append(new_t)
                for t in tracks:
                    if t not in assigned:
                        t.update(None)
            else:
                for t in tracks:
                    t.update(None)

            # Remove long-lost tracks
            tracks = [t for t in tracks if t.lost < 30]

            # ---------- Draw + Log ----------
            elapsed = time.time() - start_time
            fps = frame_id / elapsed if elapsed > 0 else 0.0
            cv2.putText(frame, f"FPS: {fps:.2f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            detections = []
            for t in tracks:
                x, y, w, h = map(int, t.bbox)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, f"ID {t.id}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                detections.append({"id": t.id, "bbox": [x, y, w, h]})

            # ---------- Log only NEW face IDs ----------
            for t in tracks:
                x, y, w, h = map(int, t.bbox)
                if t.id not in logged_ids:
                    logged_ids.add(t.id)
                    entry = {
                        "timestamp": time.time(),
                        "face_id": t.id,
                        "bbox": [x, y, w, h]
                    }
                    print(json.dumps(entry))  # still print to terminal
                    with open(LOG_PATH, "a") as f:
                        f.write(json.dumps(entry) + "\n")

            cv2.imshow("Face Detection + Kalman Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

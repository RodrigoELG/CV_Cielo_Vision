# mediapipe_test/src/main.py  
"""Webcam face detection + tracking demo (Phase 1)"""

import cv2
import mediapipe as mp
import time
import json
import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter / union if union > 0 else 0

class Track:
    _id_counter = 0
    def __init__(self, bbox):
        self.id = Track._id_counter
        Track._id_counter += 1

        # Initialize Kalman Filter: state = [cx, cy, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # State transition matrix (assuming constant velocity model)
        self.kf.F = np.array([[1., 0., 1., 0.],
                              [0., 1., 0., 1.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
        # Measurement function: we measure position only (cx, cy)
        self.kf.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])
        # Covariance matrix
        self.kf.P *= 1000.
        # Measurement uncertainty
        self.kf.R *= 10.
        # Optional: process noise Q – you can tune this if needed
        # self.kf.Q = ...

        # Initialize state
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        # Set state vector as column vector shape (4,1)
        self.kf.x = np.array([[cx],
                              [cy],
                              [0.],
                              [0.]])
        self.bbox = bbox
        self.lost = 0

    def update(self, bbox=None):
        if bbox is not None:
            # detection update
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            # Perform the predict + update step
            self.kf.predict()
            self.kf.update(np.array([[cx],
                                     [cy]]))
            self.bbox = bbox
            self.lost = 0
        else:
            # no detection: predict only
            self.kf.predict()
            self.lost += 1

        # You could update the bbox based on predicted state if you like
        # e.g., use self.kf.x to compute new bbox, but for simplicity we keep last bbox

def detect_faces(frame, face_detector):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)
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

def main():
    mp_face_detection = mp.solutions.face_detection
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    with mp_face_detection.FaceDetection(model_selection=0,
                                         min_detection_confidence=0.5) as face_detector:
        tracks = []
        frame_id = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Frame capture failed.")
                break

            frame_id += 1

            # Perform detection every N frames
            DETECT_INTERVAL = 10
            if frame_id % DETECT_INTERVAL == 0:
                boxes = detect_faces(frame, face_detector)
                assigned = []
                for box in boxes:
                    best_iou = 0.
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
                        new_track = Track(box)
                        tracks.append(new_track)
                        assigned.append(new_track)
                # For tracks that weren’t assigned this detection round, update with no bbox
                for t in tracks:
                    if t not in assigned:
                        t.update(None)
            else:
                # Just predict for all existing tracks
                for t in tracks:
                    t.update(None)

            # Clean up lost tracks
            tracks = [t for t in tracks if t.lost < 30]

            # Draw results & log
            elapsed = time.time() - start_time
            fps = frame_id / elapsed if elapsed > 0 else 0.0
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            detections = []
            for t in tracks:
                x, y, w, h = map(int, t.bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {t.id}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                detections.append({"id": t.id, "bbox": [x, y, w, h]})

            output = {
                "timestamp": time.time(),
                "frame_id": frame_id,
                "detections": detections
            }
            print(json.dumps(output))

            cv2.imshow("Webcam Face Detection + Tracking (Phase1)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

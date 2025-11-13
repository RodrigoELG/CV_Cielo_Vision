# src/frame_grabber.py
import cv2
import threading
import time
from collections import deque

class FrameGrabber(threading.Thread):
    def __init__(self, source, width, height):
        super().__init__(daemon=True)
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.latest_frame = deque(maxlen=1)
        self.running = False

    def run(self):
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.running = True
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                # optionally reconnect logic
                time.sleep(0.1)
                continue
            self.latest_frame.append(frame)
        self.cap.release()

    def stop(self):
        self.running = False

    def get_frame(self):
        if self.latest_frame:
            return self.latest_frame[-1].copy()
        else:
            return None

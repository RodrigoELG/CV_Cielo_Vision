# src/utils.py
import cv2
import numpy as np
import time

def draw_label(img, text, x, y, font_scale=0.6, thickness=1, color=(255,255,255), bg_color=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), base = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x, y-h-base-4), (x+w+10, y+base+4), bg_color, -1)
    cv2.putText(img, text, (x+5, y), font, font_scale, color, thickness, cv2.LINE_AA)

def safe_crop(frame, x, y, w, h, pad_ratio=0.2):
    H, W = frame.shape[:2]
    pad = int(pad_ratio * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return frame[y0:y1, x0:x1]

class FPSCounter:
    def __init__(self):
        self._start = time.time()
        self._frames = 0

    def tick(self):
        self._frames += 1

    def fps(self):
        dt = time.time() - self._start
        return self._frames / dt if dt > 0 else 0.0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silencia TF si está instalado
import cv2
import time
import numpy as np
from collections import deque, Counter
from statistics import mean
from insightface.app import FaceAnalysis

# ===== CONFIGURACIÓN =====
CAM_INDEX = 6 
FRAME_W, FRAME_H = 640, 480
WIN_NAME = "InsightFace (Q/ESC para salir)"
DEBUG = True

# Frecuencia de análisis (1 = todos los frames)
ANALYZE_EVERY_N_FRAMES = 1
ROLLING_WINDOW = 200
age_buffer = deque(maxlen=ROLLING_WINDOW)
gender_buffer = deque(maxlen=ROLLING_WINDOW)

# ===== UTILIDADES DE DIBUJO =====
def draw_label(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, th = 0.55, 2
    (w, h), base = cv2.getTextSize(text, font, scale, 1)
    cv2.rectangle(img, (x, y - h - 8), (x + w + 10, y + base + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 5, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

def render_stats_overlay(frame, fps, faces_count):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    genders = Counter(gender_buffer)
    total = sum(genders.values())
    avg_age = f"{mean(age_buffer):.1f}" if age_buffer else "—"
    man_pct = (100.0 * genders.get('Man', 0) / total) if total else 0.0
    wom_pct = (100.0 * genders.get('Woman', 0) / total) if total else 0.0
    lines = [
        f"FPS: {fps:.1f}",
        f"Caras (frame): {faces_count}",
        f"Detecciones: {total}",
        f"Edad promedio: {avg_age}",
        f"Genero: H {man_pct:.1f}% | M {wom_pct:.1f}%",
    ]
    x0, y0, lh = 10, 24, 22
    overlay = frame.copy()
    w_box = max(cv2.getTextSize(s, font, scale, 1)[0][0] for s in lines) + 18
    h_box = lh * len(lines) + 12
    cv2.rectangle(overlay, (x0 - 8, y0 - 20), (x0 - 8 + w_box, y0 - 20 + h_box), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    for i, s in enumerate(lines):
        cv2.putText(frame, s, (x0, y0 + i * lh), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

# ===== APP INSIGHTFACE =====
def build_insightface(det_size=(640, 640)):
    """
    Prepara FaceAnalysis intentando primero GPU (CUDAExecutionProvider)
    y si falla, cae a CPU (CPUExecutionProvider).
    """
    try:
        # providers se pasa al CONSTRUCTOR
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_size=det_size)  # 0 = GPU0
        print(">> InsightFace usando CUDAExecutionProvider (si disponible).")
        return app
    except Exception as e:
        print(">> InsightFace: fallback a CPU. Motivo:", repr(e))
        # Fuerza CPU explícitamente
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        app.prepare(ctx_id=-1, det_size=det_size)  # -1 = CPU
        return app

# ===== MAPEOS =====
def map_gender(gender_int):
    # InsightFace: gender ∈ {0,1}, donde 0 = Mujer, 1 = Hombre
    if gender_int == 1:
        return "Man"
    elif gender_int == 0:
        return "Woman"
    else:
        return None

# ===== MAIN LOOP =====
def main():
    app = build_insightface(det_size=(640, 640))

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 960, 540)

    t0 = time.time()
    frames = 0
    fps = 0.0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        idx += 1
        frames += 1
        faces_this_frame = 0

        # InsightFace espera BGR (cv2), tal como ya tenemos
        # get() hace: detección + alineamiento + atributos (age, gender si el modelo los provee)
        faces = app.get(frame)  # devuelve lista de objetos 'Face'
        if faces:
            for face in faces:
                # face.bbox = [x0, y0, x1, y1]; face.age (float), face.gender (0/1)
                x0, y0, x1, y1 = map(int, face.bbox)
                # Filtra caras pequeñas (opcional)
                if min(x1 - x0, y1 - y0) < 80:
                    continue

                faces_this_frame += 1
                # Rectángulo alrededor de la cara
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

                if idx % ANALYZE_EVERY_N_FRAMES == 0:
                    # Edad y género desde InsightFace (siempre que el pack los provea)
                    age = getattr(face, "age", None)
                    gender_raw = getattr(face, "gender", None)
                    gender_norm = map_gender(gender_raw) if gender_raw is not None else None

                    if isinstance(age, (int, float)) and age > 0:
                        age_buffer.append(float(age))
                    if gender_norm:
                        gender_buffer.append(gender_norm)

                    label_parts = []
                    if age is not None:
                        label_parts.append(f"{int(round(age))}y")
                    if gender_norm:
                        label_parts.append(gender_norm)
                    if label_parts:
                        draw_label(frame, " / ".join(label_parts), x0, max(22, y0 - 10))
                else:
                    # Si no analizamos este frame, podrías mostrar solo el score del detector,
                    # pero InsightFace no expone conf como YOLO en 'Face' por defecto.
                    pass

        # Calcula FPS cada ~0.5s
        dt = time.time() - t0
        if dt >= 0.5:
            fps = frames / dt
            frames = 0
            t0 = time.time()

        render_stats_overlay(frame, fps, faces_this_frame)
        cv2.imshow(WIN_NAME, frame)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

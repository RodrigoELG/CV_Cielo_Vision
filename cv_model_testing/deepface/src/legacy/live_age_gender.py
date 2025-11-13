#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===== CONFIGURACIÓN DE LIBRERÍAS =====
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce los logs de TensorFlow
os.environ["TQDM_DISABLE"] = "1"  # Desactiva las barras de progreso de DeepFace
import matplotlib
matplotlib.use("Agg")  # Configura matplotlib para no usar interfaz gráfica

import cv2
import time
import numpy as np
import tempfile
import traceback
from collections import deque, Counter
from statistics import mean
from ultralytics import YOLO
from deepface import DeepFace
import io
from contextlib import redirect_stdout, redirect_stderr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
YOLO_WEIGHTS = os.path.join(SRC_DIR, "weights", "yolov8n-face-lindevs.pt")

# ===== CONFIGURACIÓN DEL SISTEMA =====
# Índice de la cámara (0 = cámara principal, 6 = cámara virtual OBS)
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480  # Resolución del video
CONF_TH = 0.35  # Umbral de confianza para detección de caras
IOU_TH = 0.5   # Umbral de IoU para filtrar detecciones duplicadas
ANALYZE_EVERY_N_FRAMES = 10  # Analizar atributos cada N frames (1 = todos)
PAD_RATIO = 0.25  # Padding adicional alrededor de la cara detectada
MIN_FACE = 40     # Tamaño mínimo de cara para procesar
ACTIONS = ['age', 'gender']  # Atributos a analizar con DeepFace
WIN_NAME = "YOLOv8-face + DeepFace (Q/ESC para salir)"
DEBUG = True  # Mostrar información de debug

# ===== BUFFERS PARA ESTADÍSTICAS =====
ROLLING_WINDOW = 200  # Tamaño del buffer circular para promedios
age_buffer = deque(maxlen=ROLLING_WINDOW)     # Buffer para edades detectadas
gender_buffer = deque(maxlen=ROLLING_WINDOW)  # Buffer para géneros detectados

# ===== PRECARGA DE MODELOS =====
print(">> Precargando modelos de atributos (Age / Gender)...")
AGE_MODEL = DeepFace.build_model(model_name='Age')
GENDER_MODEL = DeepFace.build_model(model_name='Gender')

if DEBUG:
    print("   Age model:", type(AGE_MODEL).__name__)
    print("   Gender model:", type(GENDER_MODEL).__name__)

def deepface_analyze_quiet(**kwargs):
    """
    Ejecuta DeepFace.analyze sin mostrar barras de progreso ni prints.
    Redirige stdout y stderr a un buffer temporal.
    """
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        return DeepFace.analyze(**kwargs)

def draw_label(img, text, x, y):
    """
    Dibuja una etiqueta de texto con fondo negro sobre la imagen.
    Args:
        img: Imagen donde dibujar
        text: Texto a mostrar
        x, y: Coordenadas donde colocar la etiqueta
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, th = 0.55, 2
    # Calcula el tamaño del texto
    (w, h), base = cv2.getTextSize(text, font, scale, 1)
    # Dibuja el fondo negro
    cv2.rectangle(img, (x, y - h - 8), (x + w + 10, y + base + 6), (0, 0, 0), -1)
    # Dibuja el texto blanco
    cv2.putText(img, text, (x + 5, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

def render_stats_overlay(frame, fps, faces_count):
    """
    Dibuja estadísticas en tiempo real sobre el frame.
    Muestra FPS, número de caras, edad promedio y distribución de géneros.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    
    # Calcula estadísticas de género
    genders = Counter(gender_buffer)
    total = sum(genders.values())
    avg_age = f"{mean(age_buffer):.1f}" if age_buffer else "—"
    man_pct = (100.0 * genders.get('Man', 0) / total) if total else 0.0
    wom_pct = (100.0 * genders.get('Woman', 0) / total) if total else 0.0
    
    # Prepara las líneas de texto
    lines = [
        f"FPS: {fps:.1f}",
        f"Caras (frame): {faces_count}",
        f"Detecciones: {total}",
        f"Edad promedio: {avg_age}",
        f"Genero: H {man_pct:.1f}% | M {wom_pct:.1f}%",
    ]
    
    # Configuración del overlay
    x0, y0, lh = 10, 24, 22
    overlay = frame.copy()
    w_box = max(cv2.getTextSize(s, font, scale, 1)[0][0] for s in lines) + 18
    h_box = lh * len(lines) + 12
    
    # Dibuja fondo semitransparente
    cv2.rectangle(overlay, (x0 - 8, y0 - 20), (x0 - 8 + w_box, y0 - 20 + h_box), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    
    # Dibuja el texto
    for i, s in enumerate(lines):
        cv2.putText(frame, s, (x0, y0 + i * lh), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

def safe_crop(bgr, x, y, w, h, pad_ratio=0.2):
    """
    Recorta una región de la imagen de forma segura, añadiendo padding.
    Evita que el recorte se salga de los límites de la imagen.
    
    Returns:
        crop: Imagen recortada
        (x0, y0, x1, y1): Coordenadas del recorte realizado
    """
    H, W = bgr.shape[:2]
    pad = int(pad_ratio * max(w, h))  # Calcula padding proporcional
    
    # Asegura que las coordenadas estén dentro de los límites
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, W)
    y1 = min(y + h + pad, H)
    
    crop = bgr[y0:y1, x0:x1]
    return crop, (x0, y0, x1, y1)

def parse_age_gender(info: dict):
    """
    Extrae y normaliza la información de edad y género desde el resultado de DeepFace.
    Maneja diferentes formatos de respuesta y normaliza los géneros a "Man"/"Woman".
    
    Returns:
        age: Edad como número
        gender_label: Género normalizado ("Man" o "Woman")
    """
    age = info.get("age")
    gender = info.get("gender")
    dom = info.get("dominant_gender")
    
    # Determina el género dominante
    if isinstance(gender, str) and gender:
        gender_label = gender
    elif isinstance(dom, str) and dom:
        gender_label = dom
    elif isinstance(gender, dict) and gender:
        gender_label = max(gender, key=gender.get)  # Género con mayor confianza
    else:
        gender_label = None
    
    # Normaliza el género a "Man" o "Woman"
    if isinstance(gender_label, str):
        g = gender_label.lower()
        if g.startswith(("man", "male")):
            gender_label = "Man"
        elif g.startswith(("wom", "female")):
            gender_label = "Woman"
    
    return age, gender_label

def analyze_face_crop(crop_bgr):
    """
    Analiza una región de cara recortada usando DeepFace para obtener edad y género.
    Intenta primero con array numpy, si falla usa archivo temporal.
    
    Returns:
        dict: Información de edad y género, o None si falla
    """
    if crop_bgr.size == 0:
        return None
    
    try:
        # Prepara la imagen para DeepFace
        crop_bgr = np.ascontiguousarray(crop_bgr)
        if crop_bgr.dtype != np.uint8:
            crop_bgr = crop_bgr.astype(np.uint8)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        # Intenta análisis directo con array numpy
        out = deepface_analyze_quiet(
            img_path=crop_rgb,
            actions=ACTIONS,
            detector_backend="skip",  # No detectar caras, usar toda la imagen
            enforce_detection=False,
            #models={"age": AGE_MODEL, "gender": GENDER_MODEL}, 
            #prog_bar=False,
        )
        
        if isinstance(out, dict):
            return out
        elif isinstance(out, list) and out:
            return out[0]
            
    except Exception as e1:
        if DEBUG:
            print("[DeepFace ndarray] error:", repr(e1))
            traceback.print_exc()
        
        # Si falla, intenta guardando como archivo temporal
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, crop_bgr)
            
            out = deepface_analyze_quiet(
                img_path=tmp_path,
                actions=ACTIONS,
                detector_backend="skip",
                enforce_detection=False,
                # qmodels={"age": AGE_MODEL, "gender": GENDER_MODEL}, 
                # prog_bar=False,
            )
            
            if isinstance(out, dict):
                return out
            elif isinstance(out, list) and out:
                return out[0]
                
        except Exception as e2:
            if DEBUG:
                print("[DeepFace file] error:", repr(e2))
                traceback.print_exc()
        finally:
            # Limpia el archivo temporal
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    
    return None

def main():
    """
    Función principal que ejecuta el sistema de detección y análisis en tiempo real.
    Combina YOLO para detección de caras y DeepFace para análisis de atributos.
    """
    # Precarga modelos (duplicado, ya se hace al inicio)
    print(">> Precargando modelos de atributos (Age / Gender)...")
    _age = DeepFace.build_model(model_name='Age')
    _gender = DeepFace.build_model(model_name='Gender')
    
    if DEBUG:
        print("   Age model:", type(_age).__name__)
        print("   Gender model:", type(_gender).__name__)

    # Carga modelo YOLO para detección de caras
    yolo = YOLO(YOLO_WEIGHTS)


    # Inicializa cámara
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    # Configura ventana de visualización
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 960, 540)

    # Variables para cálculo de FPS
    t0 = time.time()
    frames = 0
    fps = 0.0
    idx = 0

    # ===== BUCLE PRINCIPAL =====
    while True:
        # Captura frame
        ok, frame = cap.read()
        if not ok:
            break
        
        idx += 1
        frames += 1
        
        # Detecta caras con YOLO
        results = yolo.predict(source=frame, conf=CONF_TH, iou=IOU_TH, verbose=False, device="cpu")
        faces_this_frame = 0
        
        # Procesa cada cara detectada
        if results:
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().astype(int)  # Coordenadas
                confs = r.boxes.conf.cpu().numpy()             # Confianzas
                
                for (x0, y0, x1, y1), c in zip(xyxy, confs):
                    # Convierte a formato x,y,w,h
                    x, y = max(0, x0), max(0, y0)
                    w, h = max(0, x1 - x0), max(0, y1 - y0)
                    
                    # Filtra caras muy pequeñas
                    if min(w, h) < MIN_FACE:
                        continue
                    
                    faces_this_frame += 1
                    # Dibuja rectángulo amarillo alrededor de la cara
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    
                    # Analiza atributos cada N frames
                    if idx % ANALYZE_EVERY_N_FRAMES == 0:
                        crop, _ = safe_crop(frame, x, y, w, h, PAD_RATIO)
                        info = analyze_face_crop(crop)
                        
                        if info:
                            age, gender_norm = parse_age_gender(info)
                            
                            # Actualiza buffers de estadísticas
                            if isinstance(age, (int, float)) and age > 0:
                                age_buffer.append(float(age))
                            if gender_norm:
                                gender_buffer.append(gender_norm)
                            
                            # Prepara etiqueta para mostrar
                            label = []
                            if age is not None: 
                                label.append(f"{int(round(age))}y")
                            if gender_norm: 
                                label.append(gender_norm)
                            
                            if label:
                                draw_label(frame, " / ".join(label), x, max(22, y - 10))
                        else:
                            # Si no se pudo analizar, muestra solo confianza
                            draw_label(frame, f"{c:.2f}", x, max(22, y - 10))
                    else:
                        # En frames no analizados, muestra solo confianza
                        draw_label(frame, f"{c:.2f}", x, max(22, y - 10))
        
        # Calcula FPS cada 0.5 segundos
        dt = time.time() - t0
        if dt >= 0.5:
            fps = frames / dt
            frames = 0
            t0 = time.time()
        
        # Dibuja overlay con estadísticas
        render_stats_overlay(frame, fps, faces_this_frame)
        
        # Muestra frame
        cv2.imshow(WIN_NAME, frame)
        
        # Verifica teclas de salida (ESC o Q)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q'), ord('Q')):
            break

    # Limpieza
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

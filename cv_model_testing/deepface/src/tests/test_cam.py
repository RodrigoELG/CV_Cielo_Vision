# guarda como test_cam.py y ejecuta: uv run python test_cam.py
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # o CAP_DSHOW en Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)
while True:
    ok, frame = cap.read()
    if not ok:
        print("No se pudo leer de la cámara")
        break
    cv2.imshow("TEST", frame)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q'), ord('Q')):
        break
cap.release()
cv2.destroyAllWindows()

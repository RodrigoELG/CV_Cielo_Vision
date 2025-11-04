import cv2 
import mediapipe as mp

def run_face_detection():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # For webcam input:
    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(
        model_selection=1, 
        min_detection_confidence=0.5
    ) as face_detection:
         
         while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Couldn't read camera frame")
                break

            # Convert the BGR (OpenCV) image to RGB (Mediapipe).
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            # Draw face detections of each face.
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame_rgb, detection)
            
            #Show window with detections
            cv2.imshow('MediaPipe Face Detection Test',frame_rgb)

            #Break the loop by pressing the "space" key
            if cv2.waitKey(5) & 0xFF == ord(' '):
                break 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_detection()

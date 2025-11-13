# src/attributes.py
import time
from deepface import DeepFace
from .utils import safe_crop
from .data_models import AnnotatedDetection

class AttributeAnalyzer:
    def __init__(self, analyze_interval_frames, age_model, gender_model):
        self.interval = analyze_interval_frames
        self.face_info = {}  # track_id → {age, gender, last_update_frame, timestamp}
        self.age_model = age_model
        self.gender_model = gender_model

    def analyze(self, tracked_list, frame, frame_index):
        annotated = []
        for td in tracked_list:
            info = self.face_info.get(td.track_id)
            do_analyze = False
            if info is None:
                do_analyze = True
            elif frame_index - info["last_update_frame"] >= self.interval:
                do_analyze = True

            age = None
            gender = None
            if do_analyze:
                crop = safe_crop(frame, td.bbox[0], td.bbox[1], td.bbox[2]-td.bbox[0], td.bbox[3]-td.bbox[1])
                try:
                    out = DeepFace.analyze(img_path=crop, actions=['age','gender'], detector_backend="skip",
                                            models={"age": self.age_model, "gender": self.gender_model},
                                            enforce_detection=False, prog_bar=False)
                    if isinstance(out, dict):
                        age = out.get("age")
                        gender = out.get("gender") or out.get("dominant_gender")
                except Exception as e:
                    # handle exceptions
                    pass
                self.face_info[td.track_id] = {
                    "age": age,
                    "gender": gender,
                    "last_update_frame": frame_index,
                    "timestamp": time.time()
                }
            else:
                age = info["age"]
                gender = info["gender"]

            annotated.append(AnnotatedDetection(
                bbox=td.bbox,
                confidence=td.confidence,
                track_id=td.track_id,
                age=age,
                gender=gender,
                timestamp=time.time()
            ))
        return annotated

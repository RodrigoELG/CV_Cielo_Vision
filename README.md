# CVCITUS

**Computer Vision Cielo IT U.S.**

Project focused on developing a system that uses open-source computer vision models to analyze live video feeds in real time.  
The system detects human attributes such as gender, age, range, and potentially other visual features.

---

## Repository Structure

| Folder | Description |
|--------|--------------|
| `utils/` | Helper functions and shared utilities |
| `models/` | Each subfolder contains a specific model test or experiment |
| `docs/` | Architecture diagrams, notes, and references |
| `tests/` | Unit and integration tests for project components |

---

## Current Model Tests

| Model | Description | Status |
|--------|--------------|---------|
| [MediaPipe Test](models/mediapipe_test) | Hand and face landmark detection |  Planned |
| [YOLO Age/Gender](models/yolo_age_gender) | Real-time age and gender attribute estimation |  Planned |
| [OpenVINO Optimization](models/openvino_pipeline) | Optimized inference for low-power hardware (Jetson, NUC) | Planned |

---

##  Setup Instructions

*The project runs on virtual environments to avoid diverse library version conflicts on the system. Remember to activate and exit the virtual environment whenever working on the project:

source venv/bin/activate 
deactivate 

1. Clone this repository  
   ```bash
   git clone https://github.com/RodrigoELG/CVCITUS.git
   cd CVCITUS
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Run a model test  
   ```bash
   python models/mediapipe_test/main.py
   ```

---

##  License

Still to be defined 
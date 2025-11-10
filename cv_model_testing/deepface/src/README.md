# DeepFace Live - Real-Time Facial Analysis

This project implements real-time facial analysis systems using different deep learning libraries. It includes two main implementations that detect faces and analyze attributes like age and gender from webcam input.

## 🚀 Quick Start

### Prerequisites
- Python 3.12.3
- [uv](https://docs.astral.sh/uv/) package manager
- Webcam
- CUDA 12.8+ (optional, for GPU acceleration)

### Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/SantiagoValdez/cv-age-gender-examples.git
cd cv-age-gender-examples
```

2. **Install uv** (if not already installed):
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. **Install dependencies**:
```bash
uv sync
```

4. **Download YOLO face detection model**:
Place the `yolov8n-face-lindevs.pt` model in the `weights/` directory.

5. **Run the applications**:
```bash
# YOLO + DeepFace implementation
uv run python live_age_gender.py

# InsightFace implementation  
uv run python insightface_live.py
```

### Controls
- **Q** or **ESC**: Exit the program
- Real-time statistics are displayed on screen

## 📁 Main Files

### 🎯 `live_age_gender.py` - YOLO + DeepFace

**Description**: Hybrid system that combines YOLO for face detection and DeepFace for attribute analysis.

**Models used**:
- **YOLOv8 Face**: `weights/yolov8n-face-lindevs.pt` - Fast face detection
- **DeepFace Age Model**: Pre-trained model for age estimation
- **DeepFace Gender Model**: Pre-trained model for gender classification

**Features**:
- ✅ Face detection with YOLOv8 (high speed)
- ✅ Age and gender analysis with DeepFace (high accuracy)
- ✅ Real-time statistics with circular buffers (200 samples)
- ✅ Optimized configuration to reduce logs and progress bars
- ✅ Robust error handling with fallback to temporary files
- ✅ Minimum face size filtering (80px)
- ✅ Configurable padding to improve analysis (25%)

**Configuration**:
```python
CAM_INDEX = 0                    # Main camera (6 for OBS virtual)
FRAME_W, FRAME_H = 640, 480     # Video resolution
CONF_TH = 0.35                   # YOLO confidence threshold
IOU_TH = 0.5                     # Duplicate detection filter
ANALYZE_EVERY_N_FRAMES = 1       # Analyze every N frames (1 = all)
MIN_FACE = 80                    # Minimum face size in pixels
```

**Processing flow**:
1. Capture frame from camera
2. Face detection with YOLO
3. Filter by minimum size
4. Safe cropping with padding
5. Attribute analysis with DeepFace
6. Gender normalization (Man/Woman)
7. Statistics update
8. Visualization with information overlay

### 🚀 `insightface_live.py` - Pure InsightFace

**Description**: Unified implementation using only the InsightFace library for detection and analysis.

**Models used**:
- **Buffalo_L**: Complete InsightFace model that includes:
  - Face detection
  - Facial alignment
  - Age estimation
  - Gender classification
  - Facial embeddings

**Features**:
- ✅ All-in-one solution with a single model
- ✅ Automatic GPU/CPU optimization with fallback
- ✅ Lower latency by avoiding multiple models
- ✅ Statistics similar to YOLO+DeepFace system
- ✅ Gender mapping (0=Woman, 1=Man)

**Configuration**:
```python
CAM_INDEX = 0                    # Camera index
FRAME_W, FRAME_H = 640, 480     # Resolution
ANALYZE_EVERY_N_FRAMES = 1       # Analysis frequency
det_size = (640, 640)            # Detection size
```

**Supported providers**:
- **CUDAExecutionProvider**: GPU acceleration (preferred)
- **CPUExecutionProvider**: CPU fallback

## ⚖️ Implementation Comparison

| Aspect | YOLO + DeepFace | InsightFace |
|---------|-----------------|-------------|
| **Speed** | Medium (2 models) | High (1 model) |
| **Accuracy** | High | High |
| **Memory** | Higher usage | Lower usage |
| **Flexibility** | Modular | Integrated |
| **Dependencies** | YOLO + TensorFlow | ONNX Runtime |
| **Configurability** | High | Medium |

## 🛠️ Main Dependencies

```toml
[project]
dependencies = [
    "deepface>=0.0.95",           # Facial analysis with multiple models
    "insightface>=0.7.3",         # Unified facial analysis framework
    "ultralytics>=8.2",           # YOLO for object/face detection
    "opencv-python>=4.11.0.86",   # Image and video processing
    "tensorflow[and-cuda]>=2.16", # Backend for DeepFace models
    "onnxruntime-gpu>=1.23.0",    # Optimized runtime for InsightFace
    "torch>=2.3,<3",              # PyTorch for YOLO
    "nvidia-*",                   # CUDA drivers for GPU acceleration
]
```

## � Development

### Using uv (Recommended)

uv is a fast Python package manager that handles virtual environments automatically.

```bash
# Install dependencies
uv sync

# Run scripts
uv run python live_age_gender.py
uv run python insightface_live.py

# Add new dependencies
uv add package_name

# Run with specific Python version
uv run --python 3.12 python script.py
```

### Traditional pip (Alternative)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# or
pip install -e .

# Run scripts
python live_age_gender.py
python insightface_live.py
```

## 📊 Displayed Information

Both implementations show:
- **FPS**: Real-time frames per second
- **Detected faces**: Number of faces in current frame
- **Total detections**: Cumulative counter
- **Average age**: Moving average of last 200 detections
- **Gender distribution**: Percentages of men and women

## 🎛️ Advanced Configuration

### Environment Variables:
```bash
TF_CPP_MIN_LOG_LEVEL=2    # Reduce TensorFlow logs
TQDM_DISABLE=1            # Disable progress bars
```

### Optimizations:
- **Circular buffer**: Maintains statistics for last 200 detections
- **Selective analysis**: Configurable every N frames for better performance
- **Smart padding**: Improves facial analysis quality
- **Size filtering**: Avoids processing very small faces

## 📁 Project Structure

```
deepface-live/
├── live_age_gender.py      # YOLO + DeepFace implementation
├── insightface_live.py     # InsightFace implementation
├── main.py                 # Basic main script
├── weights/                # YOLO models
│   └── yolov8n-face-lindevs.pt
├── test_cam.py            # Camera tests
├── test-gpu.py            # GPU tests
├── pyproject.toml         # Project configuration
├── uv.lock               # Dependency lock file
└── README.md             # This file
```

## 🎯 Use Cases

- **Real-time demographic analysis**
- **Age/gender-based access control**
- **Audience statistics at events**
- **Computer vision research**
- **AI application prototyping**

## ⚠️ System Requirements

- **Python 3.12.3**
- **CUDA 12.8+** (optional, for GPU acceleration)
- **Functional webcam**
- **4GB+ RAM** (8GB+ recommended for GPU)
- **Disk space**: ~2GB for models

## 🚨 Troubleshooting

### Common Issues:

1. **Camera not detected**:
   ```bash
   # Test camera
   uv run python test_cam.py
   ```

2. **GPU not working**:
   ```bash
   # Test GPU
   uv run python test-gpu.py
   ```

3. **Model download issues**:
   - Check internet connection
   - Ensure sufficient disk space
   - Clear cache: `rm -rf ~/.cache/huggingface/`

4. **Performance issues**:
   - Increase `ANALYZE_EVERY_N_FRAMES` value
   - Reduce video resolution
   - Use CPU-only mode by setting device="cpu"

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

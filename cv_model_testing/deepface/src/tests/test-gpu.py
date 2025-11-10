# --- GPU setup (TF & Torch) ---
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    if gpus:
        print(f">> TensorFlow GPU(s) visibles: {len(gpus)}")
    else:
        print(">> TensorFlow no detecta GPU")
except Exception as e:
    print(">> TensorFlow no disponible o error al configurar GPU:", repr(e))

try:
    import torch
    print(">> Torch CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(">> Torch device:", torch.cuda.get_device_name(0))
except Exception as e:
    print(">> Torch no disponible o error al consultar CUDA:", repr(e))

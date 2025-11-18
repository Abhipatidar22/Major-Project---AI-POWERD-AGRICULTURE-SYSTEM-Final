from PIL import Image
import io
import numpy as np

def load_and_preprocess(image_bytes: bytes, size=(128, 128)) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

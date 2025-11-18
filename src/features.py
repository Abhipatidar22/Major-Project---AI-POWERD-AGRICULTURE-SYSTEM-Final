from PIL import Image
import numpy as np
import json

def moisture_proxy_from_image(image: Image.Image) -> float:
    arr = np.asarray(image.convert('L')) / 255.0
    brightness = float(arr.mean())
    return 1.0 - brightness

def save_label_encoder(path, classes):
    with open(path, 'w') as f:
        json.dump({'classes': list(classes)}, f)

def load_label_encoder(path):
    with open(path, 'r') as f:
        return json.load(f)

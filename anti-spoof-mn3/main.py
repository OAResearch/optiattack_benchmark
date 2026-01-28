import os
import sys
import numpy as np
import onnxruntime as ort
import constants
from optiattack_client import collect_info

try:
    from utils import download_file
except ImportError:
    from .utils import download_file


HOST = str(sys.argv[1]) if len(sys.argv) > 1 else "localhost"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else constants.DEFAULT_CONTROLLER_PORT

def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

model_url = "https://github.com/OAResearch/optiattack_models/raw/refs/heads/main/targeted/anti-spoof-mn3.onnx"
model_path = "./model.onnx"

if not os.path.exists(model_path):
    download_file(model_url, model_path)

classes = ["real", "fake"]

# OpenVINO / CelebA-Spoof normalization (BGR)
MEAN  = np.array([151.2405, 119.5950, 107.8395], dtype=np.float32)
SCALE = np.array([63.0105, 56.4570, 55.0035], dtype=np.float32)

def preprocess_image(image):
    image = np.asarray(image)

    # Flattened input
    if image.ndim == 1:
        if image.size != 224 * 224 * 3:
            raise ValueError(f"Unexpected flattened input size: {image.size}")
        image = image.reshape(224, 224, 3)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Invalid image shape: {image.shape}")

    h, w, _ = image.shape

    # Spatial handling
    if h == 224 and w == 224:
        # Center crop (Model Expectation)
        start = (224 - 128) // 2
        image = image[start:start+128, start:start+128]
    elif h == 128 and w == 128:
        pass
    else:
        # Fallback resize (edge-case)
        from cv2 import resize, INTER_LINEAR
        image = resize(image, (128, 128), interpolation=INTER_LINEAR)

    # RGB → BGR
    image = image[..., ::-1].astype(np.float32)

    # Normalize
    image = (image - MEAN) / SCALE

    # HWC → CHW → NCHW
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image

session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

@collect_info(HOST, PORT)
def run_inference(data, additional_data=None):
    input_tensor = preprocess_image(data)

    logits = session.run(None, {input_name: input_tensor})[0]
    probs = softmax(logits, axis=1)[0]

    if additional_data is not None:
        target = additional_data.get("target")
        if target is not None and target not in classes:
            raise ValueError(f"Invalid target class: {target}")

    preds = [
        {"label": "real", "score": float(probs[0])},
        {"label": "fake", "score": float(probs[1])},
    ]

    preds.sort(key=lambda x: x["score"], reverse=True)

    return {"predictions": preds}

while True:
    pass

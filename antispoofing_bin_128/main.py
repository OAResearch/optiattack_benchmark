import os
import sys
import cv2
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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

model_url = "https://github.com/OAResearch/optiattack_models/raw/refs/heads/main/targeted/antispoofing_bin_128.onnx"
model_path = "./model.onnx"

if not os.path.exists(model_path):
    download_file(model_url, model_path)

classes = ["real", "fake"]

def preprocess_image(image):
    image = image.reshape((224, 224, 3)).astype(np.uint8)

    new_size = 128
    old_size = image.shape[:2]

    ratio = float(new_size) / max(old_size)
    scaled_shape = tuple(int(x * ratio) for x in old_size)

    image = cv2.resize(image, (scaled_shape[1], scaled_shape[0]))

    delta_w = new_size - scaled_shape[1]
    delta_h = new_size - scaled_shape[0]
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

@collect_info(HOST, PORT)
def run_inference(data, additional_data=None):
    input_tensor = preprocess_image(data)
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]

    probs = softmax(logits)

    json_results = [
        {"label": "real", "score": float(probs[0])},
        {"label": "fake", "score": float(probs[1])},
    ]
    return {"predictions": json_results}


while True:
    pass
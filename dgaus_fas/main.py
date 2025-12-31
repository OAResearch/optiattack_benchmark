import os.path
import sys

import constants
import onnxruntime as ort
import numpy as np
import requests
from optiattack_client import collect_info

try:
    from utils import download_file
except ImportError:
    from .utils import download_file

HOST = str(sys.argv[1]) if len(sys.argv) > 1 else "localhost"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else constants.DEFAULT_CONTROLLER_PORT

def softmax(x, axis):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)

model_url = "https://github.com/OAResearch/optiattack_models/raw/refs/heads/main/targeted/dgaus_fas.onnx"
model_path = "./model.onnx"

if not os.path.exists(model_path):
    download_file(model_url, model_path)

# classes
classes = ["fake", "real"]

def preprocess_image(image):
    # image: (224, 224, 3)
    image = image.reshape((224, 224, 3)).astype(np.uint8)

    # resize 224 -> 256
    from PIL import Image
    image = Image.fromarray(image)
    image = image.resize((256, 256), Image.BILINEAR)

    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))   # CHW
    image = np.expand_dims(image, axis=0)   # NCHW
    return image


session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
@collect_info(HOST, PORT)
def run_inference(data, additional_data=None):
    input_tensor = preprocess_image(data)
    # inference
    outputs = session.run(None, {input_name: input_tensor})
    output = outputs[0]
    output = softmax(output, 1)
    scores = output[0]

    if additional_data is not None:
        target_class = additional_data.get("target")
        if target_class is not None:
            if target_class not in classes:
                raise ValueError(
                    f"Target class '{target_class}' is not in the class list."
                )
            pass

    json_results = [
        {"label": "fake", "score": float(scores[0])},
        {"label": "real", "score": float(scores[1])},
    ]
    return {"predictions": json_results}

while True:
    pass
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

model_url = "https://github.com/OAResearch/optiattack_models/raw/refs/heads/main/targeted/organamnist.onnx"
model_path = "./model.onnx"

if not os.path.exists(model_path):
    download_file(model_url, model_path)

# classes
classes = [ "bladder",
            "femur-left",
            "femur-right",
            "heart",
            "kidney-left",
            "kidney-right",
            "liver",
            "lung-left",
            "lung-right",
            "pancreas",
            "spleen",]

def preprocess_image(image):
    # Eğer 1D array (150528,) geldiyse önce (224,224,3)'e reshape et
    if image.ndim == 1 and image.size == 224*224*3:
        image = image.reshape(224, 224, 3)

    # Eğer RGB ise → grayscale
    if image.ndim == 3 and image.shape[2] == 3:
        image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])  # (224,224)

    # Normalizasyon
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5

    # (H,W) → (1,1,H,W)
    image = np.expand_dims(image, axis=0)  # (1,224,224)
    image = np.expand_dims(image, axis=0)  # (1,1,224,224)

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
    indices = np.argsort(output, axis=1)[:,-5:]
    if additional_data is not None:
        target_class = additional_data.get("target")
        if target_class is not None:
            if target_class not in classes:
                raise ValueError(f"Target class '{target_class}' is not in the class list.")
            target_index = classes.index(target_class)
            found_in_top5 = target_index in indices[0]
            if not found_in_top5:
                indices[0][0] = target_index

    json_results = []
    for i in indices[0][::-1]:
        json_results.append({"label": classes[i], "score": float(output[0, i])})
    return {"predictions": json_results}

while True:
    pass
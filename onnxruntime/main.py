import onnxruntime as ort
import numpy as np
import requests
from optiattack_client import collect_info


def softmax(x, axis):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)

model_path = "resnet50-v2-7.onnx"

# classes
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = [line.strip() for line in requests.get(url).text.split("\n") if line]

def preprocess_image(image):
    image = np.reshape(image, (224,224,3))
    # Normalization
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)  # (C, H, W) -> (1, C, H, W)

    return image

session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
@collect_info()
def run_inference(data):
    input_tensor = preprocess_image(data)
    # inference
    outputs = session.run(None, {input_name: input_tensor})
    output = outputs[0]
    output = softmax(output, 1)
    indices = np.argsort(output, axis=1)[:,-5:]
    json_results = []
    for i in indices[0][::-1]:
        json_results.append({"label": classes[i], "score": float(output[0, i])})
    return {"predictions": json_results}

while True:
    pass
import onnxruntime as ort
import numpy as np
import cv2
import requests


# ONNX model yolunu belirtin
model_path = "onnxruntime.onnx"

# Sınıf isimlerini yükle
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = [line.strip() for line in requests.get(url).text.split("\n") if line]

def preprocess_image(image_path):
    """Görüntüyü yükler ve ONNX için uygun bir tensöre dönüştürür."""
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR'den RGB'ye çevir

    # Boyutlandırma ve kırpma
    h, w, _ = image.shape
    scale = 256 / min(h, w)
    resized_h, resized_w = int(h * scale), int(w * scale)
    image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # Merkezden kırpma
    start_x = (resized_w - 224) // 2
    start_y = (resized_h - 224) // 2
    image = image[start_y:start_y + 224, start_x:start_x + 224]

    # Normalizasyon
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    # Kanal sırasını değiştirme ve batch boyutunu ekleme
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)  # (C, H, W) -> (1, C, H, W)

    return image

session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
@collect_info()
def run_inference(data):
    input_tensor = preprocess_image(data)
    # Çıkarımı çalıştır
    outputs = session.run(None, {input_name: input_tensor})

    # Çıktı sınıfını belirle
    output = outputs[0]
    score = float(outputs[1])
    predicted_class = np.argmax(output)
    json_results = []
    json_results.append({"label": predicted_class, "score": score})
    return {"predictions": json_results}


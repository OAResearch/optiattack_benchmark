import base64
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.src.applications.resnet import ResNet50, preprocess_input, decode_predictions
from optiattack_client import collect_info

# Base64 string'i çöz
def decode_base64_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

# Görseli yükle ve ön işle
def preprocess_image(image):
    image = image.resize((224, 224))  # ResNet50 için uygun boyut
    image_array = np.array(image)  # NumPy array'e dönüştür
    if image_array.shape[-1] == 4:  # RGBA ise RGB'ye dönüştür
        image_array = image_array[..., :3]
    image_array = np.expand_dims(image_array, axis=0)  # Batch boyutunu ekle
    image_array = preprocess_input(image_array)  # ResNet50 preprocess işlemi
    return image_array


@collect_info()
def resnet50_main(file):

    image = decode_base64_image(file)

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions, top=10)

    json_results = []
    for imagenet_id, label, score in decoded_predictions[0]:
        json_results.append({"label": label, "score": float(score)})

    return {"predictions": json_results}

model = ResNet50(weights="imagenet")

while True:
    pass
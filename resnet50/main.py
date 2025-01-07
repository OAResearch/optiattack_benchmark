import base64
import io
import os

import numpy as np
from PIL import Image
import tensorflow as tf
from keras.applications.resnet import preprocess_input, decode_predictions, ResNet50
from optiattack_client import collect_info

def decode_base64_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

def preprocess_image(image):
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array

NET_X = 100
NET_Y = 100

@collect_info()
def resnet50_main(data):
    image_array = np.array(data, dtype=np.uint8)
    processed_image = preprocess_image(image_array)

    predictions = model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions, top=10)

    json_results = []
    for imagenet_id, label, score in decoded_predictions[0]:
        json_results.append({"label": label, "score": float(score)})

    return {"predictions": json_results}

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.experimental.list_physical_devices("GPU")

for gpu in gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
    except RuntimeError as e:
        print(e)

logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

model = ResNet50(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)
while True:
    pass
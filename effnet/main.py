import base64
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import efficientnet_b0

def decode_base64_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image):
    return transform(image)

# load classes
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = [line.strip() for line in torch.hub.load_state_dict_from_url(url).split("\n") if line]

#load pretrained model
model = efficientnet_b0(pretrained=True)
model.eval()


@collect_info()
def effnet_main(data):
    image = decode_base64_image(data)
    processed_image = preprocess_image(image)
    json_results = []
    with torch.no_grad():
        outputs = model(processed_image)
        score, predicted = outputs.max(1)  # En yüksek olasılıklı sınıfı seç
        json_results.append({"label": classes[predicted.item()], "score": float(score.item())})

    return {"predictions": json_results}



while True:
    pass
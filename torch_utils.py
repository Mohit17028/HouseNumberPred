import io
import torch
import torch.nn as nn
from inference.model_skeleton import Model
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np

# load model
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def load_model():
    device = 'cpu'
    print(os.getcwd())
    classifier = torch.load("D:\HouseNumberPred\models\SVHN_model_checkpoint.tar", map_location=device)
    inf_model = Model()
    inf_model.load_state_dict(classifier['model_state_dict'])
    inf_model.eval()
    return inf_model


model = load_model()


# image -> tensor
def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize([64, 64])
    image_array = np.array(image)
    image_array = Image.fromarray(image_array)
    t_image = test_transforms(image).view(1, 3, 64, 64)
    return t_image


# predict
def get_prediction(loaded_modal):
    image_bytes = "bd.jpg"
    image_tensor = transform_image(image_bytes=image_bytes)
    length_digits, output_digits = loaded_modal(image_tensor)
    _, length_digit = length_digits.topk(1, dim=1)
    digits_top_class = ""
    for i in range(5):
        _, _digits_top_class = output_digits[i].topk(1, dim=1)
        _digits_top_class -= 1
        if _digits_top_class == 0:
            break
        digits_top_class = digits_top_class+str(_digits_top_class)

    return length_digit, digits_top_class

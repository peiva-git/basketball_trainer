import paddle as pp

from PIL import Image
from paddleseg.models.backbones import STDC1

from .models import PPLiteSegRandomCrops


def visualize_random_crops(image: pp.Tensor, number_of_crops: int, crop_ratio: float):
    model = PPLiteSegRandomCrops(num_classes=2, backbone=STDC1(), random_crops=number_of_crops, crop_ratio=crop_ratio)
    image_batch = pp.cast(image, pp.float32)
    image_batch = pp.moveaxis(image_batch, source=-1, destination=0)
    image_batch = pp.unsqueeze(image_batch, axis=0)
    crops = model.generate_random_crops(image_batch)
    for counter, crop in enumerate(crops):
        crop = pp.squeeze(crop)
        crop = pp.moveaxis(crop, source=0, destination=-1)
        crop = pp.cast(crop, pp.uint8)
        Image.fromarray(crop.numpy()).show(title=f'Crop {counter}')

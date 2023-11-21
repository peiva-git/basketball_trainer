"""
This module contains utility functions, useful for visualization and debugging.
"""

import paddle as pp

from PIL import Image
from paddleseg.models.backbones import STDC1

from .models import PPLiteSegRandomCrops


def visualize_random_crops(image: pp.Tensor, number_of_crops: int):
    """
    Visualize all the random crops produced by the `basketballtrainer.models.pp_liteseg_rancrops.PPLiteSegRandomCrops`
    model.
    :param image: The input image fed to the model. The data format is assumed to be HWC.
    :param number_of_crops: The number of crops the model should produce for each prediction during inference
    :return: None
    """
    model = PPLiteSegRandomCrops(num_classes=2, backbone=STDC1(), random_crops=number_of_crops)
    image_batch = pp.cast(image, pp.float32)
    image_batch = pp.moveaxis(image_batch, source=-1, destination=0)
    image_batch = pp.unsqueeze(image_batch, axis=0)
    crops = model.generate_random_crops(image_batch)
    for counter, crop in enumerate(crops):
        crop = pp.squeeze(crop)
        crop = pp.moveaxis(crop, source=0, destination=-1)
        crop = pp.cast(crop, pp.uint8)
        Image.fromarray(crop.numpy()).show(title=f'Crop {counter}')

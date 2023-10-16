import random

import paddle as pp
from paddleseg.cvlibs import manager
from paddleseg.models import PPLiteSeg


@manager.MODELS.add_component
class PPLiteSegRandomCrops(PPLiteSeg):
    def __init__(self, num_classes: int, backbone, pretrained=None, random_crops: int = None):
        super().__init__(num_classes, backbone, pretrained=pretrained)
        self.__random_crops = random_crops

    def forward(self, x):
        if self.training:
            return super().forward(x)
        else:
            if self.__random_crops is not None:
                x_height_width = pp.shape(x)[2:]
                random_crops = self.__generate_random_crops(x, x_height_width[0] * 0.8, x_height_width[1] * 0.8, variance=100)
                # 1. from x, obtain random crops
                # 2. use super().forward to calculate logits for each random crop
                # 3. aggregate
                pass
            else:
                return super().forward(x)

    def __generate_random_crops(self, input_image, crop_height: int, crop_width: int, variance: int):
        image_height, image_width = pp.shape(input_image)[2:]
        crops = []
        # first crop at a random location, with the specified size
        max_x = image_width - crop_width
        max_y = image_height - crop_height
        first_crop_x = random.randint(0, max_x)
        first_crop_y = random.randint(0, max_y)
        crops.append((
            first_crop_x,
            first_crop_y,
            pp.slice(
                input_image,
                axes=(2, 3),
                starts=(first_crop_y, first_crop_x),
                ends=(first_crop_y + crop_height, first_crop_x + crop_width)
            )
        ))
        # then generate random crops similar (IoU > 0.9) to the first one
        for i in range(1, self.__random_crops):
            x = random.randint(first_crop_x - variance, first_crop_x + variance)
            y = random.randint(first_crop_y - variance, first_crop_y + variance)
            crops.append((
                max(0, x),
                max(0, y),
                pp.slice(
                    input_image,
                    axes=(2, 3),
                    starts=(max(0, y), max(0, x)),
                    ends=(min(y + crop_height, image_height), min(x + crop_width, image_width))
                )
            ))
        return crops


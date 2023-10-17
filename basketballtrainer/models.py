import random

import paddle as pp
from paddleseg.cvlibs import manager
from paddleseg.models import PPLiteSeg


@manager.MODELS.add_component
class PPLiteSegRandomCrops(PPLiteSeg):
    def __init__(self,
                 num_classes: int,
                 backbone,
                 pretrained=None,
                 random_crops: int = None,
                 crop_ratio: float = 0.8,
                 crop_variance: int = 100):
        super().__init__(num_classes, backbone, pretrained=pretrained)
        self.__random_crops = random_crops
        self.__first_crop_ratio = crop_ratio
        self.__crop_variance = crop_variance

    def forward(self, x):
        if self.training:
            return super().forward(x)
        else:
            if self.__random_crops is not None:
                x_height_width = pp.shape(x)[2:]
                # 1. from x, obtain random crops
                random_crops = self.__generate_random_crops(
                    x,
                    x_height_width[0] * self.__first_crop_ratio,
                    x_height_width[1] * self.__first_crop_ratio,
                    variance=self.__crop_variance
                )
                # 2. use super().forward to calculate logits for each random crop
                logit_tensors = [
                    # the super().forward() method generates a list of 3-D tensors, but if self.training == False
                    # that list has only one element
                    super().forward(random_crop)[0]
                    for random_crop in random_crops
                ]
                # 3. aggregate
                result = pp.mean(pp.to_tensor(logit_tensors), axis=0)
                return [result]
                pass
            else:
                return super().forward(x)

    def __generate_random_crops(self,
                                input_image: pp.Tensor,
                                crop_height: int,
                                crop_width: int,
                                variance: int) -> (int, int, pp.Tensor):
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
        for _ in range(1, self.__random_crops):
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
        # pad crops to have a constant tensor shape
        # crops should have the same shape as input_image
        return [
            pp.nn.functional.pad(
                crop,
                pad=(crop_x, image_width - crop_x - crop_width, crop_y, image_height - crop_y - crop_height),
                value=127.5
            )
            for crop_x, crop_y, crop in crops
        ]


import random

import paddle as pp
from paddleseg.models import PPLiteSeg


class PPLiteSegRandomCrops(PPLiteSeg):
    def __init__(self,
                 num_classes: int,
                 backbone,
                 backbone_indices=(2, 3, 4),
                 arm_type='UAFM_SpAtten',
                 cm_bin_sizes=(1, 2, 4),
                 cm_out_ch=128,
                 arm_out_chs=(64, 96, 128),
                 seg_head_inter_chs=(64, 64, 64),
                 resize_mode='bilinear',
                 pretrained=None,
                 random_crops: int = None,
                 crop_ratio: float = 0.8):
        super(PPLiteSegRandomCrops, self).__init__(
            num_classes=num_classes,
            backbone=backbone,
            backbone_indices=backbone_indices,
            arm_type=arm_type,
            cm_bin_sizes=cm_bin_sizes,
            cm_out_ch=cm_out_ch,
            arm_out_chs=arm_out_chs,
            seg_head_inter_chs=seg_head_inter_chs,
            resize_mode=resize_mode,
            pretrained=pretrained
        )
        self.__random_crops = random_crops
        self.__first_crop_ratio = crop_ratio

    def forward(self, x):
        """
        This method overrides the `forward` method from the `PPLiteSeg` class.
        During training, it behaves the same as the superclass method.
        During inference, it generates the specified number of random crops for each predicted image and stacks them
        by averaging their channel values.
        :param x: The input batch
        :return: A logits list
        """
        if self.training:
            return super(PPLiteSegRandomCrops, self).forward(x)
        else:
            if self.__random_crops is not None:
                # 1. from x, obtain random crops
                random_crops = self.generate_random_crops(x)
                # 2. use super().forward to calculate logits for each random crop
                logit_tensors = [
                    # the super().forward() method generates a list of 4-D tensors, but since self.training == False
                    # that list has only one element
                    super(PPLiteSegRandomCrops, self).forward(random_crop)[0]
                    for random_crop in random_crops
                ]
                # 3. aggregate
                result = pp.mean(pp.to_tensor(logit_tensors), axis=0)
                return [result]
            else:
                return super(PPLiteSegRandomCrops, self).forward(x)

    def generate_random_crops(self, input_image_batch: pp.Tensor) -> list[pp.Tensor]:
        """
        This method generates the random crops from a given batch of input images.
        All the crops have the same shape as the original image, with added padding values where needed.
        The default padding value is 127.5.
        :param input_image_batch: A tensor of shape N x C x H x W
        :return: The generated random crops, a list of N x C x H x W shaped tensors
        """
        image_height, image_width = pp.shape(input_image_batch)[2:]
        first_crop_height = int(self.__first_crop_ratio * image_height.numpy())
        first_crop_width = int(self.__first_crop_ratio * image_width.numpy())
        crops = []
        # first crop at a random location, with the specified size
        first_max_x = image_width - first_crop_width
        first_max_y = image_height - first_crop_height
        first_crop_x = random.randint(0, first_max_x)
        first_crop_y = random.randint(0, first_max_y)
        crops.append((
            first_crop_x,
            first_crop_y,
            pp.slice(
                input_image_batch,
                axes=(2, 3),
                starts=(first_crop_y, first_crop_x),
                ends=(first_crop_y + first_crop_height, first_crop_x + first_crop_width)
            )
        ))
        # then generate random crops similar (IoU > 0.9) to the first one
        variance_x = int((image_width - first_crop_width) / 4)
        variance_y = int((image_height - first_crop_height) / 4)
        for _ in range(1, self.__random_crops):
            x = random.randint(first_crop_x - variance_x, first_crop_x + variance_x)
            y = random.randint(first_crop_y - variance_y, first_crop_y + variance_y)
            crop_width = random.randint(first_crop_width - variance_x, first_crop_width + variance_x)
            crop_height = random.randint(first_crop_height - variance_y, first_crop_height + variance_y)
            crops.append((
                max(0, x),
                max(0, y),
                pp.slice(
                    input_image_batch,
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
                pad=(
                    crop_x,
                    image_width - crop_x - pp.shape(crop).numpy()[3],
                    crop_y,
                    image_height - crop_y - pp.shape(crop).numpy()[2]
                ),
                value=127.5
            )
            for crop_x, crop_y, crop in crops
        ]

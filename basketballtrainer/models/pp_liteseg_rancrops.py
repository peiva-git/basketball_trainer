"""
This module contains all the code related to the PPLiteSegRandomCrops model.
"""

import random

import paddle as pp
from paddleseg.models import PPLiteSeg


class PPLiteSegRandomCrops(PPLiteSeg):
    """
    This class represents an extension of the PPLiteSeg model.
    During training time, the PPLiteSeg model and this model behave exactly the same.
    During inference time, this model generates multiple random crops from each input image
    and computes the prediction for each of these random crops.
    All the resulting predictions are then averaged across random crops, in order to obtain a single final prediction.
    """
    __random_crops: int
    __first_crop_ratio: float

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
                 random_crops: int = None):
        """
        This constructor initializes a PPLiteSeg model with the same default parameters as the base class.
        Only the :param random_crops and :param crop_ratio parameters are specific to this model.
        For more details on the PPLiteSeg's parameters, refer to the
        [official documentation](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/paddleseg/models/pp_liteseg.py).
        :param num_classes: The number of target classes
        :param backbone: Backbone network
        :param backbone_indices: Indices of output of backbone
        :param arm_type: The type of attention refinement module
        :param cm_bin_sizes: The bin size of the context module
        :param cm_out_ch: The output channels of the last context module
        :param arm_out_chs: The output channels of each arm module
        :param seg_head_inter_chs: The intermediate channels of the segmentation head
        :param resize_mode: The resize mode for the upsampling operation in the decoder
        :param pretrained: Pretrained model path
        :param random_crops: The number of random crops to use during inference time
        The first random crop size is computed as `(crop_h, crop_w) = (img_h * crop_ratio, img_w * crop_ratio)`.
        """
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
                    # the super().forward() method generates a list of 4-D tensors
                    # it's not clear why, but the paddleseg.core.infer.inference function chooses the first element
                    super(PPLiteSegRandomCrops, self).forward(random_crop)[0]
                    for random_crop in random_crops
                ]
                # 3. aggregate
                result = pp.mean(pp.to_tensor(logit_tensors), axis=0)
                return [result]
            else:
                return super(PPLiteSegRandomCrops, self).forward(x)

    def generate_random_crops(self, input_image_batch: pp.Tensor, first_crop_ratio: float = 0.9) -> list[pp.Tensor]:
        """
        This method generates the random crops from a given batch of input images.
        All the crops have the same shape as the original image, with added padding values where needed.
        The used padding value is 0.
        :param input_image_batch: A tensor of shape N x C x H x W
        :param first_crop_ratio: Size of the first random crop, specified as a ratio of the original image size
        :return: The generated random crops, a list of N x C x H x W shaped tensors
        """
        image_height, image_width = pp.shape(input_image_batch)[2:]
        first_crop_height = int(first_crop_ratio * image_height.numpy())
        first_crop_width = int(first_crop_ratio * image_width.numpy())
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
        delta_x = int(0.05 * first_crop_width)
        delta_y = int(0.05 * first_crop_height)
        for _ in range(1, self.__random_crops):
            x = random.randint(first_crop_x - delta_x, first_crop_x + delta_x)
            y = random.randint(first_crop_y - delta_y, first_crop_y + delta_y)
            crop_width = random.randint(first_crop_width - delta_x, first_crop_width + delta_x)
            crop_height = random.randint(first_crop_height - delta_y, first_crop_height + delta_y)
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
                value=0
            )
            for crop_x, crop_y, crop in crops
        ]

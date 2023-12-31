"""
This module contains all the functions used to train the implemented models using the Command Line Interface.
"""

import argparse

from paddleseg.models import PPLiteSeg
from paddleseg.models.backbones import STDC1

from basketballtrainer.models import PPLiteSegRandomCrops
from basketballtrainer.tasks.train import train_model


def train_model_command():
    """
    This function is used as an entry point for the train command used by the `basketballtrainer` package.
    For a usage example, take a look [here](basketballtrainer.cli).
    The accepted command line arguments are `--model_type`, `--dataset_root` and `--random_crops`.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type',
        help='Choose which type of model to train',
        required=True,
        type=str,
        choices=['base', 'random-crops']
    )
    parser.add_argument(
        '--dataset_root',
        help='Set dataset root directory',
        required=True,
        type=str
    )
    parser.add_argument(
        '--random_crops',
        help='The number of random crops for random-crop-model to use during inference',
        required=False,
        type=int,
        default=2
    )

    args = parser.parse_args()
    if args.model_type == 'base':
        model = PPLiteSeg(
            num_classes=2,
            backbone=STDC1(pretrained='https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet1.tar.gz'),
            arm_out_chs=[32, 64, 128],
            seg_head_inter_chs=[32, 64, 128]
        )
        train_model(model, args.dataset_root)
    else:
        if args.random_crops < 1:
            raise ValueError(
                f'The number of random crops has to be a positive integer greater than 1, '
                f'but instead {args.random_crops} was given'
            )
        model = PPLiteSegRandomCrops(
            num_classes=2,
            backbone=STDC1(pretrained='https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet1.tar.gz'),
            arm_out_chs=[32, 64, 128],
            seg_head_inter_chs=[32, 64, 128],
            random_crops=args.random_crops
        )
        train_model(model, args.dataset_root)

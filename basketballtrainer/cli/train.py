import argparse

from paddleseg.models import PPLiteSeg
from paddleseg.models.backbones import STDC1

from basketballtrainer.models import PPLiteSegRandomCrops
from basketballtrainer.tasks.train import train_model


def train_model_command():
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
        model = PPLiteSegRandomCrops(
            num_classes=2,
            backbone=STDC1(pretrained='https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet1.tar.gz'),
            arm_out_chs=[32, 64, 128],
            seg_head_inter_chs=[32, 64, 128],
            random_crops=args.random_crops
        )
        train_model(model, args.dataset_root)

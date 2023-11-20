"""
This module contains all the validation parameters and functions.
"""

import pathlib

from paddleseg.datasets import Dataset

import paddleseg.transforms as t
from paddleseg.models import PPLiteSeg
from paddleseg.models.backbones import STDC1
from paddleseg.core import evaluate

from basketballtrainer.models import PPLiteSegRandomCrops


def __prepare_dataset(dataset_root: str):
    dataset_path = pathlib.Path(dataset_root)
    transforms = [
        t.Resize(target_size=(2048, 1024)),
        t.Normalize()
    ]
    return Dataset(
        dataset_root=str(dataset_path),
        transforms=transforms,
        num_classes=2,
        mode='val',
        val_path=str(dataset_path / 'val.txt')
    )


def evaluate_base_model(dataset_root: str, model_file: str):
    """
    This function employs the same validation parameters used in the base model configuration published in this project's
    [repository](https://github.com/peiva-git/basketball_trainer/blob/master/configs/pp_liteseg_base_stdc1_ohem_1024x512.yml).
    It is equivalent to evaluating the model using the tools provided by PaddleSeg along with the configuration file.
    :param dataset_root: Root directory of the training dataset, formatted using the [PaddleSeg specification](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.9/docs/data/custom/data_prepare.md).
    :param model_file: Model file obtained after training
    :return:
    """
    dataset = __prepare_dataset(dataset_root)
    model_path = pathlib.Path(model_file)
    model = PPLiteSeg(
        num_classes=2,
        backbone=STDC1(pretrained='https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet1.tar.gz'),
        pretrained=str(model_path),
        arm_out_chs=[32, 64, 128],
        seg_head_inter_chs=[32, 64, 128]
    )
    evaluate(
        model=model,
        eval_dataset=dataset,
        aug_eval=True,
        auc_roc=True
    )


def evaluate_rancrops_model(dataset_root: str, model_file: str, random_crops: int):
    """

    :param dataset_root:
    :param model_file:
    :param random_crops:
    :return:
    """
    dataset = __prepare_dataset(dataset_root)
    model_path = pathlib.Path(model_file)
    model = PPLiteSegRandomCrops(
        num_classes=2,
        backbone=STDC1(pretrained='https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet1.tar.gz'),
        pretrained=str(model_path),
        arm_out_chs=[32, 64, 128],
        seg_head_inter_chs=[32, 64, 128],
        random_crops=random_crops
    )
    evaluate(
        model=model,
        eval_dataset=dataset,
        aug_eval=True,
        auc_roc=True
    )

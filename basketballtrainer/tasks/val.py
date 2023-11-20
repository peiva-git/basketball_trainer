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

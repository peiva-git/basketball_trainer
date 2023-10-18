from paddleseg.models import PPLiteSeg
from paddleseg.models.backbones import STDC1
from paddleseg.models.losses import OhemCrossEntropyLoss
from paddle.optimizer.lr import PolynomialDecay
from paddle.optimizer import Momentum

import paddleseg.transforms as t
from paddleseg.datasets import Dataset
from paddleseg.core import train

from basketballtrainer.models import PPLiteSegRandomCrops


def train_model():
    model = PPLiteSeg(
        num_classes=2,
        backbone=STDC1(pretrained='https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet1.tar.gz')
    )

    train_transforms = [
        t.RandomHorizontalFlip(),
        t.Normalize()
    ]
    val_transforms = [
        t.Normalize()
    ]

    base_lr = 0.01
    lr = PolynomialDecay(base_lr, power=0.9, decay_steps=1000, end_lr=0)
    optimizer = Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)
    losses = {'types': [OhemCrossEntropyLoss(min_kept=200000)] * 3, 'coef': [1] * 3}


def train_extended_model():
    model = PPLiteSegRandomCrops(
        num_classes=2,
        backbone=STDC1(pretrained='https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet1.tar.gz'),
        random_crops=3
    )

    train_transforms = [
        t.ResizeStepScaling(min_scale_factor=0.5, max_scale_factor=2.0, scale_step_size=0.25),
        t.RandomPaddingCrop(crop_size=[1024, 512]),
        t.RandomHorizontalFlip(),
        t.RandomDistort(brightness_range=0.5, contrast_range=0.5, saturation_range=0.5),
        t.Normalize()
    ]
    val_transforms = [
        t.Normalize()
    ]

    losses = {'types': [OhemCrossEntropyLoss(min_kept=130000)] * 3, 'coef': [1] * 3}
    base_lr = 0.01
    lr = PolynomialDecay(base_lr, power=0.9, decay_steps=1000, end_lr=0)
    optimizer = Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)

    train_dataset = Dataset(
        dataset_root='/mnt/DATA/tesi/dataset/dataset_paddleseg/',
        transforms=train_transforms,
        num_classes=2,
        mode='train',
        train_path='/mnt/DATA/tesi/dataset/dataset_paddleseg/train.txt'
    )
    val_dataset = Dataset(
        dataset_root='/mnt/DATA/tesi/dataset/dataset_paddleseg/',
        transforms=val_transforms,
        num_classes=2,
        mode='val',
        val_path='/mnt/DATA/tesi/dataset/dataset_paddleseg/val.txt'
    )

    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        iters=160000,
        batch_size=4,
        use_vdl=True,
        losses=losses,
        test_config={'aug_eval': True, 'scales': 1.0, 'auc_roc': True},
        save_interval=50
    )


if __name__ == '__main__':
    train_extended_model()

import pathlib

import paddle.nn
from paddleseg.models.losses import OhemCrossEntropyLoss
from paddle.optimizer.lr import PolynomialDecay
from paddle.optimizer.lr import LinearWarmup
from paddle.optimizer import Momentum

import paddleseg.transforms as t
from paddleseg.datasets import Dataset
from paddleseg.core import train


def train_model(model: paddle.nn.Layer, dataset_root: str):
    train_transforms = [
        t.Resize(target_size=[1024, 512]),
        t.ResizeStepScaling(min_scale_factor=0.5, max_scale_factor=2.0, scale_step_size=0.25),
        t.RandomPaddingCrop(crop_size=[1024, 512]),
        t.RandomHorizontalFlip(),
        t.RandomDistort(brightness_range=0.5, contrast_range=0.5, saturation_range=0.5),
        t.Normalize()
    ]
    val_transforms = [
        t.Resize(target_size=[2048, 1024]),
        t.Normalize()
    ]

    dataset_path = pathlib.Path(dataset_root)
    train_dataset = Dataset(
        dataset_root=str(dataset_path),
        transforms=train_transforms,
        num_classes=2,
        mode='train',
        train_path=str(dataset_path / 'train.txt')
    )
    val_dataset = Dataset(
        dataset_root=str(dataset_path),
        transforms=val_transforms,
        num_classes=2,
        mode='val',
        val_path=str(dataset_path / 'val.txt')
    )

    losses = {'types': [OhemCrossEntropyLoss(min_kept=130000)] * 3, 'coef': [1] * 3}
    base_lr = 0.005
    lr = PolynomialDecay(base_lr, power=0.9, decay_steps=160000, end_lr=0)
    scheduler = LinearWarmup(lr, warmup_steps=1000, start_lr=1.0e-5, end_lr=base_lr)
    optimizer = Momentum(learning_rate=scheduler, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)

    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        iters=160000,
        batch_size=4,
        use_vdl=True,
        losses=losses,
        test_config={'aug_eval': True, 'scales': 1.0},
        save_interval=5000,
        keep_checkpoint_max=32
    )

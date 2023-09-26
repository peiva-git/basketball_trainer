from paddleseg.models import PPLiteSeg
from paddleseg.models.backbones import STDC1
from paddleseg.models.losses import OhemCrossEntropyLoss
from paddle.optimizer.lr import PolynomialDecay
from paddle.optimizer import Momentum

import paddleseg.transforms as t


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


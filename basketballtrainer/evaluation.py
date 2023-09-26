import time

import numpy as np
import paddle
from paddleseg.core import infer
from paddleseg.core.val import metrics
from paddleseg.utils import logger, TimeAverager, progbar

import paddle.nn.functional as F


def evaluate_model_with_stacking(model: paddle.nn.Layer, dataset: paddle.io.Dataset):
    # for each dataset validation image:
    # 1. generate random crops
    # 2. infer all crops
    # 3. softmax on ball channel logits
    # 4. average over all crops
    # 5. set prediction as all values above threshold
    pass


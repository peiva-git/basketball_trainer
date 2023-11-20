"""
This is the root package of the [**BasketballTrainer project**](https://github.com/peiva-git/basketball_trainer).
The main goal of this package is to train
a model for the [**BasketballDetector tool**](https://github.com/peiva-git/basketball_detector).

The model can be trained in two different ways:
1. Using the PaddleSeg tools alongside the provided
[configuration files](https://github.com/peiva-git/basketball_trainer/tree/master/configs).
A detailed description with examples is available
on [this project's public repository](https://github.com/peiva-git/basketball_trainer).
2. Using the implemented terminal commands. More details are specified in the `basketballtrainer.cli` module.

In both cases, the result of the training should be a `model.pdparams` file,
required by the `basketballdetector` package.
"""

from .tasks import train_model
from .tasks import evaluate_base_model, evaluate_rancrops_model

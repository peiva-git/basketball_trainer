"""
This package contains all the functions used to train and evaluate an instantiated model.
This functions can be used as an API and are perfectly equivalent
to training the model using the tools provided by PaddleSeg along with the
[configuration file](https://github.com/peiva-git/basketball_trainer/blob/master/configs/pp_liteseg_base_stdc1_ohem_1024x512.yml)
published in this project's repository.
"""

from .train import train_model
from .val import evaluate_base_model, evaluate_rancrops_model

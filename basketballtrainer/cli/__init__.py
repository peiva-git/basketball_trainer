"""
This package provides all the functions used by the Command Line Interface of the `basketballtrainer` package.
When the `basketballtrainer` package is installed via the `pip install basketballtrainer` command, the `train` and
`evaluate` commands are made available in the current environment.

.. include:: ./cli.md
"""

from .train import train_model_command
from .val import evaluate_model_command

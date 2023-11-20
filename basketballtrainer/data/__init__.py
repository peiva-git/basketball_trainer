"""
This package contains utilities to convert the provided dataset into various formats and to manipulate data in general.
For more information on the format this project's dataset,
take a look at the `basketballtrainer.data.convert_dataset` module.
"""

from .convert_dataset import convert_dataset_to_paddleseg_format
from .dataset_builders import PaddleSegDatasetBuilder

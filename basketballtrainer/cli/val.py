"""
This module contains all the functions used to validate the implemented models using the Command Line Interface.
"""

import argparse

from basketballtrainer import evaluate_base_model, evaluate_rancrops_model


def evaluate_model_command():
    """
    This function is used as an entry point for the train command used by the `basketballtrainer` package.
    For a usage example, take a look [here](basketballtrainer.cli).
    The accepted command line arguments are `--model_type`, `--model_file`, `--dataset_root` and `--random_crops`.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type',
        help='Choose which type of model to evaluate',
        required=True,
        type=str,
        choices=['base', 'random-crops']
    )
    parser.add_argument(
        '--model_file',
        help='The model file obtained after training',
        required=True,
        type=str
    )
    parser.add_argument(
        '--dataset_root',
        help='The root path of the evaluation dataset',
        required=True,
        type=str
    )
    parser.add_argument(
        '--random_crops',
        help='Number of random crops to use for the extended model',
        type=int,
        required=False,
        default=2
    )
    args = parser.parse_args()
    if args.model_type == 'base':
        evaluate_base_model(args.dataset_root, args.model_file)
    else:
        if args.random_crops < 1:
            raise ValueError(
                f'The number of random crops has to be a positive integer, but instead {args.random_crops} was given'
            )
        evaluate_rancrops_model(args.dataset_root, args.model_file, args.random_crops)

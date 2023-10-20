import argparse

from basketballtrainer import evaluate_base_model, evaluate_rancrops_model


def evaluate_model_command():
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
        required=False
    )
    parser.add_argument(
        '--crop_ratio',
        help='Ratio of the model\'s input image size to be used as the first random crop',
        type=float,
        required=False
    )
    args = parser.parse_args()
    if args.model_type == 'base':
        evaluate_base_model(args.dataset_root, args.model_file)
    else:
        evaluate_rancrops_model(args.dataset_root, args.model_file, args.random_crops, args.crop_ratio)

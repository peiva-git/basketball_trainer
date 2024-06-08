"""
This module contains the mask post-processing functions and is used for post-processing evaluation.
"""

import argparse
import pathlib

from skimage.io import imread_collection, imsave
from skimage.morphology import remove_small_objects

from paddleseg.utils.progbar import Progbar
from paddleseg.utils import logger
from paddleseg.utils import metrics

import numpy as np
import paddle as pp

from basketballtrainer.data import pseudocolor_mask_to_grayscale


def postprocess_mask(mask: np.ndarray, min_size: int, max_size: int) -> np.ndarray:
    """
    This function removes all the connected areas larger and smaller than the specified sizes from the provided mask.
    A new array is created, the original one is left untouched.
    :param mask: The original mask that has to be post-processed
    :param min_size: Connected areas smaller than this size will be removed
    :param max_size: Connected areas larger than this size will be removed
    :return: A new mask, where the relevant connected areas have been removed
    """
    small_removed = remove_small_objects(mask.astype(bool), min_size=min_size)
    large_objects_mask = remove_small_objects(small_removed, min_size=max_size)
    return small_removed.astype(np.int64) - large_objects_mask.astype(np.int64)


def postprocess_masks_dir(source_dir: pathlib.Path,
                          target_dir: pathlib.Path,
                          min_size: int = 625,
                          max_size: int = 900):
    """
    This function applies the `basketballtrainer.postprocessing.evaluation.postprocess_mask`
    function to all the masks in the source directory and saves the results in the target directory.
    :param source_dir: The directory where the original masks are located
    :param target_dir: The directory where the post-processed masks will be saved
    :param min_size: Connected areas smaller than this size will be removed
    :param max_size: Connected areas larger than this size will be removed
    :return: None
    """
    source_pattern = str(source_dir / '*.png')
    masks = imread_collection(source_pattern)
    for mask_index, mask in enumerate(masks):
        grayscale_mask = pseudocolor_mask_to_grayscale(mask)
        processed_mask = postprocess_mask(grayscale_mask, min_size=min_size, max_size=max_size)
        imsave(str(target_dir / f'label{mask_index + 1}.png'), processed_mask, check_contrast=False)


def evaluate_postprocessed_masks(masks_dir: pathlib.Path,
                                 ground_truths_dir: pathlib.Path,
                                 min_size: int = 625,
                                 max_size: int = 900) -> (np.ndarray, np.ndarray, np.ndarray, float):
    """
    This function applies the `basketballtrainer.postprocessing.evaluation.postprocess_mask` function
    to all the masks in the masks directory and then evaluates the IoU, Precision, Recall and Kappa  metrics
    based on the ground truth masks in the ground truth directory.
    This function assumes the masks and ground truths have the same ordering on disk,
    or, in other words, that the order is derived from the filenames.
    :param masks_dir: The directory where the masks are located
    :param ground_truths_dir: The directory where the groud truth masks are located
    :param min_size: Connected areas smaller than this size will be removed
    :param max_size: Connected areas larger than this size will be removed
    :return: The IoU, Precision, Recall and Kappa metrics
    """
    masks = imread_collection(str(masks_dir / '*.png'))
    ground_truths = imread_collection(str(ground_truths_dir / '*.png'))
    assert len(masks) == len(ground_truths), \
        f'Should have the same number of masks and ground truths, ' \
        f'but got {len(masks)} masks and {len(ground_truths)} ground truths'

    progbar = Progbar(target=len(masks), verbose=1)
    logger.info(f'Start evaluating (total_samples: {len(masks)})')

    intersect_area_all = pp.zeros([1], dtype='int64')
    pred_area_all = pp.zeros([1], dtype='int64')
    label_area_all = pp.zeros([1], dtype='int64')

    for index, (mask, ground_truth) in enumerate(zip(masks, ground_truths)):
        grayscale_mask = pseudocolor_mask_to_grayscale(mask)
        filtered_mask = postprocess_mask(grayscale_mask, min_size=min_size, max_size=max_size)
        intersect_area, pred_area, label_area = metrics.calculate_area(
            pp.to_tensor(filtered_mask, dtype='int64'),
            pp.to_tensor(ground_truth, dtype='int64'),
            2,
            ignore_index=255
        )
        intersect_area_all = intersect_area_all + intersect_area
        pred_area_all = pred_area_all + pred_area
        label_area_all = label_area_all + label_area

        progbar.update(index + 1)

    metrics_input = (intersect_area_all, pred_area_all, label_area_all)
    class_iou, _ = metrics.mean_iou(*metrics_input)
    _, class_precision, class_recall = metrics.class_measurement(*metrics_input)
    kappa = metrics.kappa(*metrics_input)

    logger.info(f'Class IoU: {np.round(class_iou, 4)}')
    logger.info(f'Class Precision: {np.round(class_precision, 4)}')
    logger.info(f'Class Recall: {np.round(class_recall, 4)}')
    logger.info(f'Kappa: {kappa:.4f}')

    return class_iou, class_precision, class_recall, kappa


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--masks_dir',
        help='Root directory of the masks generated by the model',
        type=str,
        required=True
    )
    parser.add_argument(
        '--gt_dir',
        help='Root directory of the corresponding ground truths',
        type=str,
        required=True
    )
    parser.add_argument(
        '--min_size',
        help='Objects smaller than this size will be filtered out of the mask',
        type=int,
        required=False,
        default=625  # 25 * 25
    )
    parser.add_argument(
        '--max_size',
        help="Objects larger than this size will be filtered out of the mask",
        type=int,
        required=False,
        default=900  # 30 * 30
    )
    args = parser.parse_args()
    evaluate_postprocessed_masks(pathlib.Path(args.masks_dir),
                                 pathlib.Path(args.gt_dir),
                                 min_size=args.min_size,
                                 max_size=args.max_size)

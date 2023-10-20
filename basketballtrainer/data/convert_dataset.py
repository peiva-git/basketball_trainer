"""
This module contains utility functions to convert the provided dataset into various formats.
This function assumes the provided dataset has the following structure, starting from the dataset's root:
1. The [season]/[match]/frames/ directory contains the video frames
2. The [season]/[match]/masks/ directory contains the corresponding ground truth masks

The same game may be played in different seasons. The correspondences between frames and masks are determined by
the frame and mask indexes (the last integer in the filename).
**Please note** that this script won't work if the filenames or the directory structure differ from the specification.
"""

import argparse
import glob
import os.path
import pathlib
import shutil


def convert_dataset_to_paddleseg_format(dataset_path: str, target_path: str):
    """
    Copy images and labels annotated in MATLAB to PaddleSeg-compliant directory structure.
    **Please note** that the dataset structure generated by this function still has to be split and shuffled by following the
    instructions found [here](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/data/marker/marker.md#4-split-a-custom-dataset)
    :param dataset_path: path string pointing to the root of the MATLAB dataset
    :param target_path: root of the newly created PaddleSeg dataset
    """
    source = pathlib.Path(dataset_path)
    target = pathlib.Path(target_path)
    images, labels = __generate_ordered_filenames_lists(source)
    if not os.path.exists(str(target / 'images')):
        os.mkdir(str(target / 'images'))
    if not os.path.exists(str(target / 'labels')):
        os.mkdir(str(target / 'labels'))

    for sample_index in range(len(images)):
        shutil.copy2(images[sample_index], str(target / f'images/image{sample_index + 1}.png'))
        shutil.copy2(labels[sample_index], str(target / f'labels/label{sample_index + 1}.png'))


def __generate_ordered_filenames_lists(source: pathlib.Path) -> ([str], [str]):
    images = []
    labels = []
    for match_directory_path in glob.iglob(str(source / '*/*')):
        match_directory = pathlib.Path(match_directory_path)
        match_image_paths = [
            match_image_path
            for match_image_path in glob.iglob(str(match_directory / 'frames/*.png'))
        ]
        match_mask_paths = [
            match_mask_path
            for match_mask_path in glob.iglob(str(match_directory / 'masks/*.png'))
        ]
        match_image_paths.sort(key=lambda file_path: int(file_path.split('_')[-1].split('.')[-2]))
        match_mask_paths.sort(key=lambda file_path: int(file_path.split('_')[-1].split('.')[-2]))
        images.extend(match_image_paths)
        labels.extend(match_mask_paths)
    return images, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_dir',
        help='Root directory of the original dataset',
        type=str,
        required=True
    )
    parser.add_argument(
        '--target_dir',
        help='Target directory to store the new generated dataset',
        type=str,
        required=True
    )
    args = parser.parse_args()
    convert_dataset_to_paddleseg_format(args.source_dir, args.target_dir)

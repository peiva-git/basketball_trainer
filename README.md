# Basketball Detector Trainer

[![build and deploy docs](https://github.com/peiva-git/basketball_trainer/actions/workflows/docs.yml/badge.svg)](https://github.com/peiva-git/basketball_trainer/actions/workflows/docs.yml)
[![build and test CPU](https://github.com/peiva-git/basketball_trainer/actions/workflows/build-and-test-cpu.yml/badge.svg)](https://github.com/peiva-git/basketball_trainer/actions/workflows/build-and-test-cpu.yml)
[![build and test GPU](https://github.com/peiva-git/basketball_trainer/actions/workflows/build-and-test-gpu.yml/badge.svg)](https://github.com/peiva-git/basketball_trainer/actions/workflows/build-and-test-gpu.yml)
![License](https://img.shields.io/github/license/peiva-git/basketball_trainer)


This repository contains all the necessary tools to train the
:basketball: [BasketballDetector](https://github.com/peiva-git/basketball_detector) model.

## Table Of Contents

1. [Description](#description)
2. [Using the PaddleSeg toolbox](#using-the-paddleseg-toolbox)
3. [Environment setup](#environment-setup)
4. [Model training](#model-training)
5. [Model evaluation](#model-evaluation)
6. [Results](#results)
   1. [OHEM Cross-Entropy loss function results](#ohem-cross-entropy-loss-function-results)
   2. [Weighted Cross-Entropy loss function results](#weighted-cross-entropy-loss-function-results)
   3. [Rancrop model results](#rancrop-model-results)
7. [Credits](#credits)

## Description

This project uses the [PaddleSeg toolkit](https://github.com/PaddlePaddle/PaddleSeg)
to train a modified [PPLiteSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.8/configs/pp_liteseg)
real-time semantic segmentation model.
The configuration files used during training can be found [here](configs).
In the following sections, you will find detailed instructions on how to set up a working environment and
how to train a model.

## Using the PaddleSeg toolbox

The segmentation model has been trained using a customized version of the sample
configuration file for the PPLiteSeg model applied to the
[Cityscapes dataset](https://www.cityscapes-dataset.com/) found
[on the PaddleSeg repository](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k.yml).

## Environment setup

Before being able to train the model, you must install [Paddle](https://github.com/PaddlePaddle/Paddle) and
[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg).
You can use one of the [provided conda environments](conda) by running the following command:
```shell
conda create --name myenv-pp --file pp-[cpu|gpu].yml
```
It is recommended to have a CUDA enabled GPU in order to take advantage of GPU acceleration.

In case you're using the GPU version, don't forget to set up the required environment variables as well:
```shell
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
```
You can automate the process of adding the environment variables to execute automatically 
each time you activate your conda environment by running the following commands:
```shell
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

The currently available [Paddle PyPi GPU version release](https://pypi.org/project/paddlepaddle-gpu/) requires the
CUDA Runtime API version 10.2 to be installed in order to run correctly.
This dependency is therefore listed in the provided [conda environment](conda/pp-gpu.yml).
**If you want to use a different CUDA version**, refer to the
[official documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html).

Also, to avoid unexpected errors, the [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
package should be built from source using the provided repository,
while being in the myenv-pp environment:
```shell
cd PaddleSeg
pip install -v -e .
```

## Model training

To train the BasketballDetector segmentation model, run:
```shell
python PaddleSeg/tools/train.py \
--configs configs/pp_liteseg_base_stdc1_ohem_10000_1024x512.yml \
--do_eval \
--use_vdl \
--save_interval 2500 \
--keep_checkpoint_max 20 \
--save_dir output
```
The trained models will then be available in the `output/` directory.
More information on what these options do and on how to visualize the training process
can be found [here](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/train/train.md).

## Model evaluation

To evaluate the obtained model, run:
```shell
python PaddleSeg/tools/val.py \
--configs configs/pp_liteseg_base_stdc1_ohem_10000_1024x512.yml \
--model_path output/best_model/model.pdparams
```

For additional options refer to the
[official documentation](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/evaluation/evaluate.md).

## Results

Various models have been trained using two different loss functions,
in particular the
[OHEM Cross-Entropy loss function](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.9/docs/apis/losses/losses.md#ohemcrossentropyloss)
and the
[weighted Cross-Entropy loss function](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.9/docs/apis/losses/losses.md#crossentropyloss).
All the used configurations with different parameters can be found [here](configs).

In the following tables you can find the summarized results.

### OHEM Cross-Entropy loss function results

| `min_kept` parameter value     | Ball class IoU | Ball class Precision | Ball class Recall | Kappa      | Links                                                                         |
|--------------------------------|----------------|----------------------|-------------------|------------|-------------------------------------------------------------------------------|
| 40 x 40 x `batch_size` = 6400  | 0,6590         | 0,8527               | **0,7437**        | 0,7944     | [config](configs/ohem_variants/pp_liteseg_base_stdc1_ohem_6400_1024x512.yml)  |
| 50 x 50 x `batch_size` = 10000 | **0,6658**     | 0,8770               | 0,7344            | **0,7993** | [config](configs/ohem_variants/pp_liteseg_base_stdc1_ohem_10000_1024x512.yml) |
| 60 x 60 x `batch_size` = 14400 | 0,6601         | 0,8895               | 0,7191            | 0,7952     | [config](configs/ohem_variants/pp_liteseg_base_stdc1_ohem_14400_1024x512.yml) |
| 70 x 70 x `batch_size` = 20000 | 0,6542         | **0,9090**           | 0,7000            | 0,7979     | [config](configs/ohem_variants/pp_liteseg_base_stdc1_ohem_20000_1024x512.yml) |

### Weighted Cross-Entropy loss function results

| Background class weight | Ball class IoU | Ball class precision | Ball class Recall | Kappa      | Links                                                                       |
|-------------------------|----------------|----------------------|-------------------|------------|-----------------------------------------------------------------------------|
| 0,001                   | 0,2656         | 0,2700               | **0,9422**        | 0,4195     | [config](configs/wce_variants/pp_liteseg_base_stdc1_wce_0.001_1024x512.yml) |
| 0,005                   | 0,4703         | 0,4927               | 0,9117            | 0,6396     | [config](configs/wce_variants/pp_liteseg_base_stdc1_wce_0.005_1024x512.yml) |
| 0,01                    | **0,5394**     | **0,5729**           | 0,9020            | **0,7007** | [config](configs/wce_variants/pp_liteseg_base_stdc1_wce_0.01_1024x512.yml)  |

### Rancrop model results

The `PPLiteSegRandomCrops` model was validated using
[these configurations](configs/rancrop_ohem_10000) and [these configurations](configs/rancrop_wce_0.01),
with the model described in [this configuration](configs/pp_liteseg_base_stdc1_ohem_10000_1024x512.yml)
used as the base model.

Various variants of the `PPLiteSegRandomCrops` model were tested, all available in [this directory](rancrop_model_variants).
In particular, different padding and aggregation methods were used.
All the obtained results were worse than the chosen base PPLiteSeg model.

## Credits

This project uses the [PaddleSeg toolbox](https://github.com/PaddlePaddle/PaddleSeg). All credits go to its authors.
This project uses [pdoc](https://pdoc.dev/) to generate its documentation. All credits go to its authors.
The implemented `PPLiteSegRandomCrops` model takes inspiration from the paper
[Real-time CNN-based Segmentation Architecture for Ball Detection in a Single View Setup](https://arxiv.org/abs/2007.11876).
All credits go to its authors.

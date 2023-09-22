# Basketball Detector Trainer

This repository contains all the necessary tools to train the
:basketball:[BasketballDetector](https://github.com/peiva-git/basketball_detector) model.

## Table Of Contents

1. [Description](#description)
2. [Using the PaddleSeg toolbox](#using-the-paddleseg-toolbox)
3. [Environment setup](#environment-setup)
4. [Model training](#model-training)
5. [Model evaluation](#model-evaluation)
6. [Results](#results)
7. [Credits](#credits)

## Description

This project uses the [PaddleSeg toolkit](https://github.com/PaddlePaddle/PaddleSeg)
to train a [PPLiteSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.8/configs/pp_liteseg)
real-time semantic segmentation model.
The configuration files used during training can be found [here](config).
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
You can use the [provided conda environment](conda/pp-environment.yml) by running the following command:
```shell
conda create --name myenv-pp --file pp-environment.yml
```

**Please note** that both the provided environment and the
[Paddle PyPi release](https://pypi.org/project/paddlepaddle-gpu/) currently 
require the CUDA Runtime API version 10.2 to be installed in order to run correctly.
If you want a different version, refer to the 
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
cd PaddleSeg
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
--config ../config/pp_liteseg_stdc1_basketballdetector_1024x512_pretrain-10rancrops.yml \
--do_eval \
--use_vdl \
--save_interval 500
```
The trained models will then be available in the `PaddleSeg/output` directory.
More information on what these options do and on how to visualize the training process
can be found [here](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/train/train.md).

## Model evaluation

To evaluate the obtained model, run:
```shell
cd PaddleSeg
python tools/val.py \
--config ../basketballdetector/config/pp_liteseg_stdc1_basketballdetector_1024x512_pretrain-10rancrops.yml \
--model_path output/best_model/model.pdparams
```

For additional options refer to the
[official documentation](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/evaluation/evaluate.md).

## Results

The following results have been obtained by training a model with 
[this configuration](config/pp_liteseg_stdc1_basketballdetector_1024x512_pretrain-10rancrops.yml)
using the tools provided by [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/train/train.md).

In the following table you can find the summarized results of the obtained model.
Most of the columns are self-explanatory, aside from:
1. Train Random Crops: number of random crops transformations applied to each sample during training.
Since the model's postprocessing leverages multiple heatmaps to obtain better results, a comparison has been made
2. Pretrained Backbone: whether the model uses a backbone pretrained on the cityscapes dataset or not.
In the latter case, using a pretrained backbone isn't possible since a custom number of input channels is used instead.

| Model        | Backbone | Train Random Crops | Pretrained Backbone | Train  Resolution | Test  Resolution | Training Iters | mIoU   | Ball Class IoU | Links                |
|--------------|----------|--------------------|---------------------|-------------------|------------------|----------------|--------|----------------|----------------------|
| PP-LiteSeg-T | STDC1    | 1                  | Yes                 | 1024x512          | 2048x1024        | 160000         | 0.8232 | 0.6466         | config model log vdl |
| PP-LiteSeg-T | STDC1    | 10                 | Yes                 | 1024x512          | 2048x1024        | 160000         |        |                | config model log vdl |
| PP-LiteSeg-T | STDC1    | 1                  | No                  | 1024x512          | 2048x1024        | 160000         |        |                | config model log vdl | 
| PP-LiteSeg-T | STDC1    | 10                 | No                  | 1024x512          | 2048x1024        | 160000         |        |                | config model log vdl |

## Credits

This project uses the [PaddleSeg toolbox](https://github.com/PaddlePaddle/PaddleSeg). All credits go to its authors.
This project uses [pdoc](https://pdoc.dev/) to generate its documentation. All credits go to its authors.

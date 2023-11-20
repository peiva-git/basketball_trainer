The `train` command accepts the following options:

| Parameter    | Description                                                                                                    | Is Required | Default value |
|--------------|----------------------------------------------------------------------------------------------------------------|-------------|---------------|
| model_type   | Choose which model to train, base (PPLiteSeg) or random-crops (basketballtrainer.models.PPLiteSegRandomCrops). | Yes         | -             |
| dataset_root | Root directory of the dataset used during training                                                             | Yes         | -             |
| random_crops | The number of random crops for the random-crops model to use during inference                                  | No          | 2             |

The `evaluate` command accepts the following parameters:

| Parameter    | Description                                                                                                       | Is Required | Default value |
|--------------|-------------------------------------------------------------------------------------------------------------------|-------------|---------------|
| model_type   | Choose which model to evaluate, base (PPLiteSeg) or random-crops (basketballtrainer.models.PPLiteSegRandomCrops). | Yes         | -             |
| model_file   | The path to the model file obtained after training                                                                | Yes         | -             |
| dataset_root | Root directory of the dataset used during validation                                                              | Yes         | -             |
| random_crops | The number of random crops for the random-crops model to use during inference                                     | No          | 2             |

For example, for training:
```shell
train \
--model_type base \
--dataset_root /mnt/data/dataset
```

And for evaluation:
```shell
evalutate \
--model_type base \
--model_file /home/ubuntu/model.pdparams \
--dataset_root /mnt/data/dataset
```

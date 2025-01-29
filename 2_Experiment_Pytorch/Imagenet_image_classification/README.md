# Image Classification on ILSVRC2012

## Introduction

This repo is modified form the [project](https://github.com/rwightman/pytorch-image-models).

The bash file demonstrates 4 GPU (GTX 2080TI) training the image classification model with Frankenstein optimizer and achieving State-of-the-Art.


## Getting started

```
bash run_resnet50.sh # for Frankenstein on resnet50
bash run_resnet18.sh # for Frankenstein on resnet18
```

## Result

Top 1 testing accuracy:
|     Model     | accuracy |
| ------------- | -------- |
|   Resnet18    | 71.2512  |
|   Resnet50    | 77.5576  |

<img width="800" alt="portfolio_view" src="https://github.com/acctouhou/Frankenstein_optimizer_temp/blob/main/2_Experiment_Pytorch/Imagenet_image_classification/imagenet.png">

The raw classification results for ResNet18 and ResNet50 using five different random seeds can be found in the results directory.
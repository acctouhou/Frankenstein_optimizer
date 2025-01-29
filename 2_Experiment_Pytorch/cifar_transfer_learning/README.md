# PyTorch CIFAR Training

## Overview
This project provides an implementation for transfer learning training CIFAR-10 and CIFAR-100 datasets using PyTorch

## Features
- Supports training on CIFAR-10 and CIFAR-100 datasets.
- Multiple deep learning models: ResNet, DenseNet, VGG, and EfficientNet.
- Various optimization algorithms, including:
  - SGD
  - Adam
  - AMSGrad
  - RAdam
  - Lookahead with Adam (la_adam)
  - AdaBelief
  - AdaBound
  - Frankenstein Optimizer (Our)
  - Ranger (Lookahead with RAdam)
  - RMSprop

## Usage
### Training a Model
Run the script with the following command:
```sh
python train.py --model eff --optim frank --total_epoch 50 --batchsize 128
```

### Available Arguments
- `--total_epoch`: Total number of training epochs (default: 50)
- `--model`: Model architecture to use (`resnet`, `densenet`, `vgg`, `eff` for EfficientNet)
- `--optim`: Optimizer to use (e.g., `adam`, `sgd`, `radam`, `frank`, etc.)
- `--batchsize`: Batch size for training and testing (default: 128)
- `--lr`: Initial learning rate (default: 1e-4)
- `--momentum`: Momentum term for applicable optimizers (default: 0.9)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--beta1`: Adam optimizer beta1 coefficient (default: 0.9)
- `--beta2`: Adam optimizer beta2 coefficient (default: 0.999)
- `--final_lr`: Final learning rate for AdaBound (default: 0.1)
- `--gamma`: Convergence speed term for AdaBound (default: 1e-3)
- `--pretrain`: Whether to use pretrained weights (default: True)
- `--resume`: Resume training from checkpoint
- `--size`: EfficientNet model size index (default: 0)

## Model Selection
The script supports various architectures:
- **EfficientNet**: Models ranging from EfficientNet-B0 to EfficientNet-B6.

Example to train with EfficientNet-B4 and Frankenstein optimizer:
```
python train.py --model eff --size 4 --optim frank --batchsize 128 --total_epoch 50
```

## Learning Rate Scheduling
The learning rate can be adjusted dynamically using a step scheduler. By default, the learning rate decreases at defined epochs.

## Logging and Saving Results
Logging and Saving Results

Training logs are saved in text files, following this format:

ft_100_frank_4.txt: Trained on CIFAR-100 with Frankenstein optimizer and EfficientNet-B4.

ft_10_frank_0.txt: Trained on CIFAR-10 with Frankenstein optimizer and EfficientNet-B0.

ft_100_sgd_2.txt: Trained on CIFAR-100 with SGD optimizer and EfficientNet-B2.

Each log file contains training and testing accuracies for each epoch.
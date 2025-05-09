# Model-Agnostic Meta-Learning on omniglot

## Introduction

This repo is modified form the [project](https://github.com/tristandeleu/pytorch-maml).


In this framework, optimizers listed below are available:
- **0.Frankenstein**
- **1.Adam**
- **2.AMSGrad**
- **3.SGD**
- **4.RMSprop**
- **5.Lookahead(Adam)**
- **6.AdaBound**
- **7.Ranger**


## Getting started

```
python main.py data --taropt 0 --output-folder temp --dataset omniglot --num-ways 5 --num-shots 20 --use-cuda --step-size 0.1 --batch-size 64 --num-workers 2 --num-epochs 100 --meta-lr 2e-4 --output-folder to/results&

#taropt: the target optimizer id 0 for Frankenstein
```

## Result
<img width="800" alt="portfolio_view" src="https://github.com/acctouhou/Frankenstein_optimizer_temp/blob/main/2_Experiment_Pytorch/MAML/maml.png">

The following result files follow the naming convention x_y_z, where x represents the index of the optimizer type, y corresponds to the num-ways parameter in the MAML algorithm, and z denotes the num-shots parameter.








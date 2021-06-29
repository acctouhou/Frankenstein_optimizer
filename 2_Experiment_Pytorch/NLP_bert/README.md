# Bidirectional Encoder Representations from Transformers(BERT) with imdb text classification train on vary optimizers

## Introduction

This repo is modified form the [project](https://github.com/huggingface/transformers).

The scripts perform training text classification with different optimizers,including pre-trained and random initial  weights.


In this framework, optimizers listed below are available:
- **0.Frankenstein**
- **1.Adam**
- **2.AMSGrad**
- **3.SGD**
- **4.RMSprop**
- **5.Lookahead(Adam)**
- **6.AdaBound**
- **7.Ranger**


## Run on code

```
python transfer.py 0 0    #training as the fine-tune
#first variable is GPU ID
#second variable is target optimizers id

python inital.py 0 0    #training as random initial 
#first variable is GPU ID
#second variable is target optimizers id

```

## Result  

### The testing accuracy with pre-trained weights

<img width="800" alt="portfolio_view" src="https://github.com/acctouhou/Frankenstein_optimizer_temp/blob/main/2_Experiment_Pytorch/NLP_bert/transfer_bert.png">

### The testing accuracy with random initial  weights

<img width="800" alt="portfolio_view" src="https://github.com/acctouhou/Frankenstein_optimizer_temp/blob/main/2_Experiment_Pytorch/NLP_bert/initial_bert.png">

#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py "$@" data --model resnet50 --batch-size 64 --weight-decay 1e-2 --sched step --lr 1e-3 --warmup-epochs 5 --epochs 120 --opt Frankenstein --min-lr 1e-5 --decay-epochs 40 --amp --warmup-lr 1e-6 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --mixup 0.2 --mixup-off-epoch 80

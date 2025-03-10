#!/usr/bin/env bash

# bash build.sh

# pip install setuptools==59.5.0

ulimit -c unlimited

today=`date +%Y%m%d_%T_log`
touch ../log/$today.txt

CUDA_VISIBLE_DEVICES=$1 fairseq-train \
--user-dir ../tokengt \
--num-workers 16 \
--ddp-backend=legacy_ddp \
--dataset-name pcqm4mv2 \
--dataset-source ogb \
--task graph_prediction \
--criterion l1_loss \
--arch tokengt_base \
--lap-node-id \
--lap-node-id-k 16 \
--lap-node-id-sign-flip \
--lap-node-id-eig-dropout 0.2 \
--stochastic-depth \
--prenorm \
--num-classes 1 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.1 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
--lr 2e-4 --end-learning-rate 1e-9 \
--batch-size 128 \
--fp16 \
--data-buffer-size 20 \
--save-dir ./ckpts/pcqv2-tokengt-lap16 \
--tensorboard-logdir ./tb/pcqv2-tokengt-lap16 \
--no-epoch-checkpoints\
> ../log/$today.txt
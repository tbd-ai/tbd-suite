#!/bin/bash

BS=4
APP_single_gpu="python train_mem_profile.py \
--train-manifest /SCRATCH/dataset/librispeech/libri_train_manifest.csv.new.new.new \
--val-manifest /SCRATCH/dataset/librispeech/libri_val_manifest.csv.new.new.new \
--opt-level O1 \
--num-workers 8 \
--batch-size ${BS} \
--epochs 1 \
--log-dir log_mem_prof/ "

APP_multi_gpus="python -m torchelastic.distributed.launch \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=4 \
        train_mem_profile.py --train-manifest /SCRATCH/dataset/librispeech/libri_train_manifest.csv.new.new.new \
                 --val-manifest /SCRATCH/dataset/librispeech/libri_val_manifest.csv.new.new.new \
                 --opt-level O1 \
                 --num-workers 8 \
                 --batch-size ${BS} \
                 --epochs 1 \
                 --log-dir log_mem_prof/ "
$(APP_single_gpu)

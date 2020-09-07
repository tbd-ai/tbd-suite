#!/bin/bash
cd /source/deepspeech.pytorch;
Train_Manifest_Path='/scratch/dataset/librispeech/libri_train_manifest.csv.new'
Val_Manifest_Path='/scratch/dataset/librispeech/libri_val_manifest.csv.new' 
BATCH_SIZE=8

python -m torchelastic.distributed.launch \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=4 \
        train.py --train-manifest ${Train_Manifest_Path} \
                 --val-manifest ${Val_Manifest_Path} \
                 --opt-level O1 \
                 --num-workers 8 \
                 --batch-size ${BATCH_SIZE} \
                 --epochs 30 \
                 --checkpoint \
                 --tensorboard \
                 --log-dir log_dir/ \
                 --save-n-recent-models 3
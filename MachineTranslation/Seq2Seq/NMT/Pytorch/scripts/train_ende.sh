#! /usr/bin/env bash
set -e

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

DATASET_DIR=/dataset/wmt16_de_en
LOG_DIR=/scratch/train_log
RESULT_DIR=/scratch

rm /source/GNMT/seq2seq/train/trainer.py;
ln -s /source/GNMT/seq2seq/train/trainer.py.noprofile /source/GNMT/seq2seq/train/trainer.py;

cd /source/GNMT;
python3 -m launch train.py --dataset-dir ${DATASET_DIR} \
 	--seed 2 \
 	--train-global-batch-size 1024 \
 	--train-batch-size 64 \
 	--val-batch-size 32 \
 	--test-batch-size 64 \
 	--results-dir=${RESULT_DIR} \
 	--math fp32


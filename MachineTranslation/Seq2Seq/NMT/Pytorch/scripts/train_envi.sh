#! /usr/bin/env bash
set -e

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

DATASET_DIR=/dataset/iwslt_en_vi
LOG_DIR=/scratch/train_vi_log
RESULT_DIR=/scratch/results

rm /source/GNMT/seq2seq/train/trainer.py;
ln -s /source/GNMT/seq2seq/train/trainer.py.noprofile /source/GNMT/seq2seq/train/trainer.py;

cd /source/GNMT;

python3 -m launch train.py --dataset-dir ${DATASET_DIR} \
	--seed 2 \
	--epochs 30 \
	--src-lang en --tgt-lang vi \
	--train-src ${DATASET_DIR}/train-preproc.bpe.32000.en \
	--train-tgt ${DATASET_DIR}/train-preproc.bpe.32000.vi \
	--val-src ${DATASET_DIR}/tst2012.bpe.32000.en \
	--val-tgt ${DATASET_DIR}/tst2012.bpe.32000.vi \
	--test-src ${DATASET_DIR}/tst2013.bpe.32000.en \
	--test-tgt ${DATASET_DIR}/tst2013.bpe.32000.vi \
	--train-global-batch-size 512 \
	--train-batch-size 64 \
	--val-batch-size 32 \
	--test-batch-size 64 \
	--results-dir=${RESULT_DIR} \
	--math fp32 | tee -a $LOG_DIR


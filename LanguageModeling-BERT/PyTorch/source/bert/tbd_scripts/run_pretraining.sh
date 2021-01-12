#!/bin/bash

mkdir -p /workspace/bert/results
DATADIR=/workspace/bert/data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/

TRAIN_SCRIPT=/workspace/bert/run_pretraining.py

NUM_GPUs=1
BATCH_SIZE=${1:-16}
START_ITER=${2:--1}
STOP_ITER=${3:--1}
PROFILE=${4:-}


echo starting training
cmd="python3 -m torch.distributed.launch \
	--nproc_per_node=$NUM_GPUs \
	$TRAIN_SCRIPT \
	--input_dir=$DATADIR \
	--output_dir=/workspace/bert/results/checkpoints \
	--config_file=/workspace/bert/bert_config.json \
	--bert_model=bert-large-uncased \
	--train_batch_size=$BATCH_SIZE \
	--max_seq_length=128 \
	--max_predictions_per_seq=20 \
	--disable_progress_bar \
	--max_steps=7038 \
	--warmup_proportion=0.2843 \
	--num_steps_per_checkpoint=20000 \
	--learning_rate=6e-3 \
	--seed=42 \
	--fp16 \
	--gradient_accumulation_steps=1 \
	--allreduce_post_accumulation \
	--allreduce_post_accumulation_fp16 \
	--do_train \
	--benchmark_start=$START_ITER \
	--benchmark_stop=$STOP_ITER \
	--json-summary /workspace/bert/results/dllogger.json \
	$PROFILE"

echo $cmd
eval $cmd

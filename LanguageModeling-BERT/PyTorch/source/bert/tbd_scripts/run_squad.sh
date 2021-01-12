#!/bin/bash


TRAIN_SCRIPT=/workspace/bert/run_squad.py

TRAIN_FILE=/workspace/bert/data/download/squad/v1.1/train-v1.1.json
PREDICT_FILE=/workspace/bert/data/download/squad/v1.1/dev-v1.1.json

EVAL_SCRIPT=/workspace/bert/data/download/squad/v1.1/evaluate-v1.1.py

OUTPUT_DIR=/workspace/bert/results/SQuAD

VOCAB_FILE=/workspace/bert/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt

LOG_FILE=/workspace/bert/results/SQuAD/logfile.txt

mkdir -p /workspace/bert/results/SQuAD/
touch $LOG_FILE

NUM_GPUs=4
BATCH_SIZE=${1:-16}
STOP_ITER=${2:--1}


python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUs \
        $TRAIN_SCRIPT \
	--init_checkpoint=/scratch/bert_uncased.pt  \
	--do_train  \
	--train_file=$TRAIN_FILE \
	--train_batch_size=$BATCH_SIZE \
	--do_predict \
	--predict_file=$PREDICT_FILE \
	--predict_batch_size=4 \
	--eval_script=$EVAL_SCRIPT \
    --do_eval  \
	--do_lower_case \
    --bert_model=bert-large-uncased \
    --learning_rate=3e-5  \
	--seed=1 \
	--fp16 \
	--log_freq=10 \
    --max_steps=$STOP_ITER \
    --num_train_epochs=3  \
	--max_seq_length=384  \
	--doc_stride=128 \
    --disable-progress-bar \
    --output_dir=/workspace/bert/results/SQuAD  \
	--vocab_file=$VOCAB_FILE \
    --config_file=/workspace/bert/bert_config.json  \
	--json-summary=/workspace/bert/results/SQuAD/dllogger.json  \
	|& tee $LOG_FILE

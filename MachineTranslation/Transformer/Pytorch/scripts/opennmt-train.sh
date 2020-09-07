#!/bin/bash
DATASET_DIR=/dataset/train.atok.low
OUTPUT_DIR=/scratch

cd /source/OpenNMT-py;
python3  train.py -data $DATASET_DIR -save_model $OUTPUT_DIR \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
        -world_size 4 -gpu_ranks 0 1 2 3

# python  train.py -data /tmp/de2/data -save_model /tmp/extra -gpuid 1 \
#         -layers 6 -rnn_size 512 -word_vec_size 512 -batch_type tokens -batch_size 4096 \
#         -epochs 50  -max_generator_batches 32 -normalization tokens -dropout 0.1 -accum_count 4 \
#         -max_grad_norm 0 -optim adam -encoder_type transformer -decoder_type transformer \
#         -position_encoding -param_init 0 -warmup_steps 16000 -learning_rate 2 -param_init_glorot \
#         -start_checkpoint_at 5 -decay_method noam -label_smoothing 0.1 -adam_beta2 0.998 -report_every 1000

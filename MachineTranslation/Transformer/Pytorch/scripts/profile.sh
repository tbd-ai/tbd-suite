#!/bin/bash
NSIGHT_PREFIX="/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --csv --target-processes all --profile-from-start no --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_fp16.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"

DATASET_DIR=/dataset/train.atok.low
OUTPUT_DIR=/scratch

export CUDA_VISIBLE_DEVICES=0

cd /source/OpenNMT-py;


for BATCH_SIZE in 2 4 8 16 32 64 128
do
	echo Profiling batch size ${BATCH_SIZE}

	if [ $1 == "nvprof" ]; then
		rm /source/OpenNMT-py/onmt/trainer.py;
		ln -s /source/OpenNMT-py/onmt/trainer.py.profile_nvprof /source/OpenNMT-py/onmt/trainer.py;

		NVPROF_PREFIX="nvprof --profile-from-start off --export-profile /scratch/measurements/profile_`date +%s`_bs${BATCH_SIZE}.nvvp"

		$NVPROF_PREFIX python3 train.py -data $DATASET_DIR -save_model $OUTPUT_DIR \
			-layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
			-encoder_type transformer -decoder_type transformer -position_encoding \
			-train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
			-batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
			-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
			-max_grad_norm 0 -param_init 0  -param_init_glorot \
			-label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
			-world_size 4 -gpu_ranks 0
	elif [ $1 == "nsight" ]; then
		rm /source/OpenNMT-py/onmt/trainer.py;
		ln -s /source/OpenNMT-py/onmt/trainer.py.profile_nsight /source/OpenNMT-py/onmt/trainer.py;

		$NSIGHT_PREFIX python3 train.py -data $DATASET_DIR -save_model $OUTPUT_DIR \
			-layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
			-encoder_type transformer -decoder_type transformer -position_encoding \
			-train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
			-batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
			-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
			-max_grad_norm 0 -param_init 0  -param_init_glorot \
			-label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
			-world_size 4 -gpu_ranks 0 | tee -a /scratch/measurements/nsight_bs${BATCH_SIZE}.log
	fi
done

# python  train.py -data /tmp/de2/data -save_model /tmp/extra -gpuid 1 \
#         -layers 6 -rnn_size 512 -word_vec_size 512 -batch_type tokens -batch_size 4096 \
#         -epochs 50  -max_generator_batches 32 -normalization tokens -dropout 0.1 -accum_count 4 \
#         -max_grad_norm 0 -optim adam -encoder_type transformer -decoder_type transformer \
#         -position_encoding -param_init 0 -warmup_steps 16000 -learning_rate 2 -param_init_glorot \
#         -start_checkpoint_at 5 -decay_method noam -label_smoothing 0.1 -adam_beta2 0.998 -report_every 1000

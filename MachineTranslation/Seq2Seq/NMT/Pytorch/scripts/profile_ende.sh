#! /usr/bin/env bash
set -e

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

NSIGHT_PREFIX="/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --target-processes all --csv --profile-from-start no --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_fp16.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"

DATASET_DIR=/dataset/wmt16_de_en
LOG_DIR=/scratch/train_log
RESULT_DIR=/scratch/results

export LD_LIBRARY_PATH=/usr/local/lib/python3.6/dist-packages/torch/lib/:$LD_LIBRARY_PATH

cd /source/GNMT;

for BATCH_SIZE in 2 4 8 16 32 64 128
do
	if [ $1 == "nvprof" ]; then
		rm /source/GNMT/seq2seq/train/trainer.py;
		ln -s /source/GNMT/seq2seq/train/trainer.py.profile_nvprof /source/GNMT/seq2seq/train/trainer.py;

		NVPROF_PREFIX="nvprof --profile-from-start off --export-profile /scratch/measurements/profile_`date +%s`_bs${BATCH_SIZE}.nvvp"

		$NVPROF_PREFIX python3 -m launch train.py --dataset-dir ${DATASET_DIR} \
			--seed 2 \
			--train-global-batch-size 1024 \
			--train-batch-size $BATCH_SIZE \
			--val-batch-size 32 \
			--test-batch-size 64 \
			--results-dir=${RESULT_DIR} \
			--math fp32
	elif [ $1 == "nsight" ]; then
		rm /source/GNMT/seq2seq/train/trainer.py;
		ln -s /source/GNMT/seq2seq/train/trainer.py.profile_nsight /source/GNMT/seq2seq/train/trainer.py;

		$NSIGHT_PREFIX python3 -m launch train.py --dataset-dir ${DATASET_DIR} \
			--seed 2 \
			--train-global-batch-size 1024 \
			--train-batch-size $BATCH_SIZE \
			--val-batch-size 32 \
			--test-batch-size 64 \
			--results-dir=${RESULT_DIR} \
			--math fp32 | tee -a /scratch/nsight_bs${BATCH_SIZE}.log

	elif [ $1 == "memory" ]; then
		rm /source/GNMT/seq2seq/train/trainer.py;
		ln -s /source/GNMT/seq2seq/train/trainer.py.profile_memory /source/GNMT/seq2seq/train/trainer.py;

		$NVPROF_PREFIX python3 -m launch train.py --dataset-dir ${DATASET_DIR} \
			--seed 2 \
			--train-global-batch-size 1024 \
			--train-batch-size $BATCH_SIZE \
			--val-batch-size 32 \
			--test-batch-size 64 \
			--results-dir=${RESULT_DIR} \
			--math fp32
	fi
done


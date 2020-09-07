#! /usr/bin/env bash
set -e

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

NSIGHT_PREFIX="/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --target-processes all --csv --profile-from-start no --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_fp16.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"

DATASET_DIR=/dataset/iwslt_en_vi
LOG_DIR=/scratch/train_vi_log
RESULT_DIR=/scratch/results
# --train-global-batch-size 512 \

rm /source/GNMT/seq2seq/train/trainer.py;
ln -s /source/GNMT/seq2seq/train/trainer.py.profile /source/GNMT/seq2seq/train/trainer.py;


cd /source/GNMT;

for BATCH_SIZE in 2 4 8 16 32 64 128
do
	echo Testing with batch size $BATCH_SIZE
	if [ $1 == "nvprof" ]; then
		NVPROF_PREFIX="nvprof --profile-from-start off --export-profile /scratch/measurements/profile_`date +%s`_bs${BATCH_SIZE}.nvvp"
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
			--train-batch-size $BATCH_SIZE \
			--val-batch-size 32 \
			--test-batch-size 64 \
			--results-dir=${RESULT_DIR} \
			--math fp32
	elif [ $1 == "nsight" ]; then
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
			--train-batch-size $BATCH_SIZE \
			--val-batch-size 32 \
			--test-batch-size 64 \
			--results-dir=${RESULT_DIR} \
			--math fp32 | tee -a /scratch/nsight_bs${BATCH_SIZE}.log
	fi
done


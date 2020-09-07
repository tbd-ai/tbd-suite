export PYTHONIOENCODING=UTF-8

NSIGHT_PREFIX="/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --csv --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_fp16.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"

mkdir /scratch/measurements;

cd /source/nmt;
export CUDA_VISIBLE_DEVICES=0;

for BATCH_SIZE in 32 64 `seq 128 128 1024`
do
	echo Testing batch size $BATCH_SIZE

	if [ $1 == "nvprof" ]; then
		NVPROF_PREFIX="nvprof --profile-from-start off --export-profile /scratch/measurements/profile_`date +%s`_bs${BATCH_SIZE}.nvvp"
		echo $NVPROF_PREFIX

		$NVPROF_PREFIX python3 -m nmt.nmt \
		    --batch_size=$BATCH_SIZE \
		    --num_train_steps=1000 \
		    --src=vi --tgt=en \
		    --vocab_prefix=/dataset/vocab  \
		    --train_prefix=/dataset/train \
		    --dev_prefix=/dataset/tst2012  \
		    --test_prefix=/dataset/tst2013 \
		    --out_dir=/scratch \
		    --hparams_path=/source/hparams/iwslt15_profiling.json

	elif [ $1 == "nsight" ]; then
		$NSIGHT_PREFIX python3 -m nmt.nmt \
		    --batch_size=$BATCH_SIZE \
		    --num_train_steps=1000 \
		    --src=vi --tgt=en \
		    --vocab_prefix=/dataset/vocab  \
		    --train_prefix=/dataset/train \
		    --dev_prefix=/dataset/tst2012  \
		    --test_prefix=/dataset/tst2013 \
		    --out_dir=/scratch \
		    --hparams_path=/source/hparams/iwslt15_profiling.json | tee -a /scratch/measurements/nsight_bs${BATCH_SIZE}.log
	fi
done


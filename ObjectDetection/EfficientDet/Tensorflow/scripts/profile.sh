cd /source/automl/efficientdet; 

rm det_model_fn.py;
ln -s det_model_fn.py.profile det_model_fn.py;

# begin training
DATASET_DIR=/dataset

DATASET=coco
NUM_EPOCHS=100

export CUDA_VISIBLE_DEVICES=0

for BATCH_SIZE in 1 2 4 8 16 32
do
	NVPROF_PREFIX="nvprof --profile-from-start off --export-profile /scratch/measurements/profile_`date +%s`_bs${BATCH_SIZE}.nvvp"
	NSIGHT_PREFIX="/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --csv --target-processes all --profile-from-start no --metrics sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,gpc__cycles_elapsed.max"

	if [ "$1" == "nvprof" ]; then
		rm testhook.py
		ln -s testhook.py.profile_nvprof testhook.py

		$NVPROF_PREFIX python3 main.py --mode=train \
		    --training_file_pattern=${DATASET_DIR}/tfrecord/train*.tfrecord \
		    --validation_file_pattern=${DATASET_DIR}/tfrecord/val*.tfrecord \
		    --val_json_file=${DATASET_DIR}/annotations/instances_val2017.json \
		    --model_name=efficientdet-d0 \
		    --model_dir=/tmp/efficientdet-d0-scratch  \
		    --backbone_ckpt=efficientnet-b0  \
		    --train_batch_size=$BATCH_SIZE \
		    --eval_batch_size=8 --eval_samples=512 \
		    --num_examples_per_epoch=5717 --num_epochs=${NUM_EPOCHS}  \
		    --hparams="num_classes=20,moving_average_decay=0" \
		    --use_tpu=False 
	elif [ "$1" == "nsight" ]; then
		rm testhook.py
		ln -s testhook.py.profile_nsight testhook.py

		$NSIGHT_PREFIX python3 main.py --mode=train \
		    --training_file_pattern=${DATASET_DIR}/tfrecord/train*.tfrecord \
		    --validation_file_pattern=${DATASET_DIR}/tfrecord/val*.tfrecord \
		    --val_json_file=${DATASET_DIR}/annotations/instances_val2017.json \
		    --model_name=efficientdet-d0 \
		    --model_dir=/tmp/efficientdet-d0-scratch  \
		    --backbone_ckpt=efficientnet-b0  \
		    --train_batch_size=$BATCH_SIZE \
		    --eval_batch_size=8 --eval_samples=512 \
		    --num_examples_per_epoch=5717 --num_epochs=${NUM_EPOCHS}  \
		    --hparams="num_classes=20,moving_average_decay=0" \
		    --use_tpu=False | tee -a /scratch/measurements/nsight_bs${BATCH_SIZE}.log
	fi
done

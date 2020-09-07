cd /source/EfficientDet.Pytorch;

NSIGHT_PREFIX="/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --csv --target-processes all --profile-from-start no --metrics sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,gpc__cycles_elapsed.max"

export LD_LIBRARY_PATH=/root/.local/lib/python3.6/site-packages/torch/lib/:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

for BATCH_SIZE in 1 4 8 16
do
	echo Running batch size $BATCH_SIZE
	if [ "$1" == "nsight" ]; then
		rm train.py;
		ln -s train.py.profile_nsight train.py;
		$NSIGHT_PREFIX python3 train.py.profile \
			--dataset COCO \
			--dataset_root /dataset \
			--save_folder /scratch \
			--network efficientdet-d0 \
			--num_epoch 1 \
			--batch_size $BATCH_SIZE | tee -a /scratch/measurements/bs${BATCH_SIZE}.log
	elif [ "$1" == "nvprof" ]; then
		NVPROF_PREFIX="nvprof --profile-from-start off --export-profile /scratch/measurements/profile_`date +%s`_bs${BATCH_SIZE}.nvvp"
		rm train.py;
		ln -s train.py.profile_nvprof train.py;
		$NVPROF_PREFIX python3 train.py.profile \
			--dataset COCO \
			--dataset_root /dataset \
			--save_folder /scratch \
			--network efficientdet-d0 \
			--num_epoch 1 \
			--batch_size $BATCH_SIZE
	fi
done

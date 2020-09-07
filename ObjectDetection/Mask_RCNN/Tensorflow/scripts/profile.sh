cd /source/Mask_RCNN/samples/coco;

export CUDA_VISIBLE_DEVICES=0
NSIGHT_PREFIX="/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --csv --target-processes all --profile-from-start no --metrics sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,gpc__cycles_elapsed.max"

for BATCH_SIZE in 4 8 16
do
	echo Using batch size $BATCH_SIZE

	if [ "$1" == "nsight" ]; then
		$NSIGHT_PREFIX python3 coco.py train --dataset=/dataset --model=coco --download=true | tee -a /scratch/measurements/nsight_bs${BATCH_SIZE}.log
	elif [ "$1" == "nvprof" ]; then
		NVPROF_PREFIX="nvprof --profile-from-start off --export-profile /scratch/measurements/nvprof_`date +%s`_bs${BATCH_SIZE}.nvvp -f --print-summary"
		$NVPROF_PREFIX python3 coco.py train --dataset=/dataset --model=coco --download=true 
	fi
done


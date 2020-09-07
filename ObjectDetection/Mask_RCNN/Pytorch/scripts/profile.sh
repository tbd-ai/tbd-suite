cd /source/detectron2_repo/tools;

NSIGHT_PREFIX="/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --csv --target-processes all --profile-from-start no --metrics sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,gpc__cycles_elapsed.max"

export DETECTRON2_DATASETS=/dataset/
export LD_LIBRARY_PATH=/root/.local/lib/python3.6/site-packages/torch/lib/:$LD_LIBRARY_PATH

for BATCH_SIZE in 4 8 16
do
	echo Running batch size $BATCH_SIZE
	if [ "$1" == "nvprof" ]; then
		NVPROF_PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off --export-profile /scratch/measurements/profile_`date +%s`_bs${BATCH_SIZE}.nvvp -f --print-summary"
		$NVPROF_PREFIX python3 ./train_net.py --num-gpus 1 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml TEST.EVAL_PERIOD 1000 SOLVER.IMS_PER_BATCH $BATCH_SIZE SOLVER.MAX_ITER 1000 
	elif [ "$1" == "nsight" ]; then
		$NSIGHT_PREFIX python3 ./train_net.py --num-gpus 1 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml TEST.EVAL_PERIOD 1000 SOLVER.IMS_PER_BATCH $BATCH_SIZE SOLVER.MAX_ITER 1000 | tee -a /scratch/measurements/nsight_bs${BATCH_SIZE}.log
	fi
done

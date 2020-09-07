cd /source/detectron2_repo/tools;

NVPROF_PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off --export-profile /mnt/profiler_output.nvvp -f --print-summary"
NSIGHT_PREFIX="/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --csv --target-processes all --profile-from-start no --metrics sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,gpc__cycles_elapsed.max"

# DETECTRON2_DATASETS=/mnt/dataset/ python3 ./train_net.py --num-gpus 4 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml | tee -a /mnt/scripts/log

export DETECTRON2_DATASETS=/dataset/
export LD_LIBRARY_PATH=/root/.local/lib/python3.6/site-packages/torch/lib/:$LD_LIBRARY_PATH

for BATCH_SIZE in 4
do
	echo Running batch size $BATCH_SIZE
	python3 ./train_net.py --num-gpus 1 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml TEST.EVAL_PERIOD 1000 SOLVER.IMS_PER_BATCH $BATCH_SIZE SOLVER.MAX_ITER 1000 | tee -a /scratch/mem.log
done

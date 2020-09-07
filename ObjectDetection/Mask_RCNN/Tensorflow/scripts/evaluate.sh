NSIGHT_PREFIX="/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --csv --profile-from-start off -o profile --metrics all"
NVPROF_PREFIX="/usr/local/cuda/bin/nvprof -o profiler_output_eval.nvvp -f -metrics all"
cd /mnt/source/new_test/Mask_RCNN/samples/coco;
python3 coco.py evaluate --dataset=/mnt/dataset/voc --model=coco


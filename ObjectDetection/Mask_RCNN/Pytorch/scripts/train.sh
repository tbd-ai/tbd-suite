cd /source/detectron2_repo/tools;

export DETECTRON2_DATASETS=/dataset/
export LD_LIBRARY_PATH=/root/.local/lib/python3.6/site-packages/torch/lib/:$LD_LIBRARY_PATH

# normal run
python3 ./train_net.py --num-gpus 4 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml TEST.EVAL_PERIOD 1000 SOLVER.MAX_ITER 10000

# memory profiling
# python3 ./train_net.py --num-gpus 4 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml SOLVER.MAX_ITER 200

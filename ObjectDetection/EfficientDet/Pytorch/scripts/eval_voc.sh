WEIGHTS_ROOT=/mnt/source/EfficientDet.Pytorch/saved/weights/COCO/efficientdet-d0

cd /source/EfficientDet.Pytorch;
# python3 eval.py --dataset COCO --dataset_root /mnt/dataset --weight ${WEIGHTS_ROOT}/checkpoint_45.pth
# python3 eval.py --dataset COCO --dataset_root /dataset --weight /source/EfficientDet.Pytorch/saved/weights/COCO/efficientdet-d0/checkpoint_68.pth
python3 eval.py --dataset VOC --dataset_root /dataset/VOCdevkit --weight /scratch/checkpoint_VOC_efficientdet-d0_268.pth

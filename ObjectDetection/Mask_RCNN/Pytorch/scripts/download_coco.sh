#!/bin/bash

cd /dataset

wget https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# download weights
wget https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl

unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
tar xvf *.tgz


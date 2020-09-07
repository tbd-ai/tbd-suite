#!/bin/bash

DATA_DIR=/home/ybgao/DNN-Training-Suite/MachineTranslation-Transformer/Pytorch_1/dataset/new-data
if [ ! -z "$1" ]
then
	DATA_DIR=$1
	if [ ! -d "$DATA_DIR" ]
	then
		echo "Supplied data directory $DATA_DIR does not exist"
		exit -1
	fi
fi
echo "Using dataset from $DATA_DIR"

docker run -it --runtime=nvidia \
	-v ${DATA_DIR}:/dataset \
	-v ${PWD}/source:/source \
	-v ${PWD}/scripts:/scripts \
	-v ${PWD}/scratch:/scratch \
	--name tbd-transformer-pytorch-exp --privileged --shm-size=32G  tbd-transformer-pytorch

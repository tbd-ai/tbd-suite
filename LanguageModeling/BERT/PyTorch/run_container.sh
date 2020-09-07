#!/bin/bash

DATA_DIR=/home/SCRATCH/dataset/tbd-bert-wiki-bookcorpus-pytorch/data
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

docker run -it --runtime=nvidia -v ${DATA_DIR}:/workspace/bert/data \
    -v ${PWD}/scratch:/scratch/  -v ${PWD}/source:/workspace \
    --name tbd-bert-pytorch-exp --privileged --shm-size=32G  tbd-bert-pytorch

#!/bin/bash

DATA_DIR=/home/SCRATCH/dataset/imagenet/
if [ ! -z "$1" ]
then
    DATA_DIR=$1
    if [ ! -d "$DATA_DIR" ]
    then
        echo "Supplied data directory $DATA_DIR does not exist"
        exit -1
    fi
    echo "Using dataset from $DATA_DIR"
fi

docker run -it --runtime=nvidia -v $PWD/source:/app/source -v ${DATA_DIR}:/app/data \
    --name tbd-resnet-tf-exp --shm-size=32G  --privileged  tbd-resnet-tf

#!/bin/bash

if [ ! -z "$1" ]
then
    DATA_DIR=$1
    if [ ! -d "$DATA_DIR" ]
    then
        echo "Supplied data directory $DATA_DIR does not exist"
        exit -1
    fi
    echo "Using dataset from $DATA_DIR"
else
    echo "Positional argument <data dir> is required"
    exit -1
fi


docker run -it --runtime=nvidia -v ${DATA_DIR}:/app/data  -v $PWD/source:/app/source \
    --name tbd-inception_v3-pytorch-exp --shm-size=32G  --privileged tbd-inception_v3-pytorch

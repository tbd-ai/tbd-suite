#!/bin/bash
if [ ! -f "./docker/Dockerfile" ]; then
   echo "File ./docker/Dockerfile dese not exist, make sure you invoke the script in the right directory"
   exit 1
fi

docker build -t tbd-resnet-mxnet --build-arg host_uid=$UID ./docker

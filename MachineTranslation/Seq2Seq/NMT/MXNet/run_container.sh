#!/bin/bash

docker run -it --runtime=nvidia \
	-v ${PWD}/source:/source \
	-v ${PWD}/scripts:/scripts \
	-v ${PWD}/scratch:/scratch \
	--name tbd-nmt-mxnet-exp --privileged --shm-size=32G  tbd-nmt-mxnet

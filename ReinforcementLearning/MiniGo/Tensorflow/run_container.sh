#!/bin/bash
docker run -it --runtime=nvidia \
	-v ${DATA_DIR}:/dataset \
	-v ${PWD}/source:/source \
	-v ${PWD}/scripts:/scripts \
	-v ${PWD}/scratch:/scratch \
	--name tbd-minigo-tf-exp --privileged --shm-size=40G  tbd-minigo-tf
#!/bin/bash

pushd /app/source/train > /dev/null
dataset=/app/data/
export PYTHONPATH=$PYTHONPATH:/app/source/train/common
# simply refer to the arguments in the script
COMMAND="python3 /app/source/train/train_imagenet.py  --data-train /app/data/imagenet1k_train_240.rec $*"
echo $COMMAND
eval $COMMAND

popd > /dev/null

#!/bin/bash

pushd `dirname "$0"` > /dev/null
dataset=/app/data/


# simply refer to the arguments in the script
COMMAND="python ../main.py -a inception_v3 $dataset $*"

echo $COMMAND
eval $COMMAND

popd > /dev/null

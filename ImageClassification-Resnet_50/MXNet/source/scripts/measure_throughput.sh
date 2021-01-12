#!/bin/bash

# training configuration
start_iteration=${1:-1000}
end_iteration=${2:-2000}

export PYTHONPATH=/app/source/train/common:$PYTHONPATH
pushd `dirname "$0"` > /dev/null

dataset=/app/data/imagenet1k_train_240.rec

result_dir="./throughput_data"

if [ -d "$result_dir" ]
then
    echo "Cannot create result file, directory $result_dir already exists"
    exit -1
fi

mkdir $result_dir

for batch_size in 1 2 4 8 16 32 64 96 128
do
    # simply refer to the arguments in the script
    COMMAND="python3 /app/source/train/train_imagenet.py  --data-train $dataset --batch-size  $batch_size --nvprof-start-batch $start_iteration --nvprof-stop-batch $end_iteration --only-throughput --nvprof-on"
    COMMAND="$COMMAND |& tee $result_dir/batch_$batch_size.log"
    echo $COMMAND
    eval $COMMAND
    sleep 5
done
popd > /dev/null

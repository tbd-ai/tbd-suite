#!/bin/bash

# training configuration
start_iteration=${1:-1000}
end_iteration=${2:-2000}

pushd `dirname "$0"` > /dev/null

dataset=/app/data/

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
    COMMAND="python ../main.py -a inception_v3 $dataset -b  $batch_size --profile_warmup $start_iteration --profile_finish $end_iteration"
    COMMAND="$COMMAND | grep 'Batch throughput' | tee $result_dir/batch_$batch_size.log"
    echo $COMMAND
    eval $COMMAND
    sleep 5
done
popd > /dev/null

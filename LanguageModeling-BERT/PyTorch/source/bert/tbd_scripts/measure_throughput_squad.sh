#!/bin/bash

# training configuration
end_iteration=${1:-2000}

pushd `dirname "$0"` > /dev/null

dataset=/app/data/

result_dir="./throughput_data_squad"

if [ -d "$result_dir" ]
then
    echo "Cannot create result file, directory $result_dir already exists"
    exit -1
fi

mkdir $result_dir

for batch_size in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    # simply refer to the arguments in the script
    COMMAND="bash /workspace/bert/tbd_scripts/run_squad.sh $batch_size $end_iteration"
    COMMAND="$COMMAND | grep 'running throughput average' | tee $result_dir/batch_$batch_size.log"
    echo $COMMAND
    eval $COMMAND
    sleep 5
done
popd > /dev/null

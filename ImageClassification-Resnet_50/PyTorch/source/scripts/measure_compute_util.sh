#!/bin/bash
# training configuration
profiler=${1:-'nvprof'}
start_iteration=${2:-1000}
end_iteration=${3:-2000}

result_dir="./profiled_data_$profiler"

if [ -d "$result_dir" ]
then
    echo "Cannot create result file, directory $result_dir already exists"
    exit -1
fi

mkdir $result_dir

if [ $profiler != 'nvprof' ] && [ $profiler != 'ncu' ]
then
    echo "Profiler (first argument) must be either nvprof or cnu"
    exit -1
fi

pushd `dirname "$0"` > /dev/null
dataset=/app/data/

AWK_CODE='{if(start==1) print($0); if (match($0, "==PROF== Disconnected from process")) start=1; }'

for batch_size in 1 2 4 8 16 32 64 96 128
do
    if [ $profiler = 'nvprof' ]
    then
        prefix="$profiler --profile-from-start off --export-profile  $result_dir/compute_utilization_batch_${batch_size}_%p.nvvp --print-summary"
    else
        prefix="ncu --csv --target-processes all --profile-from-start no --metrics sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,gpc__cycles_elapsed.max"
    fi
    # simply refer to the arguments in the script
    COMMAND="$prefix python ../main.py -a resnet50 $dataset -b  $batch_size --profile_warmup $start_iteration --profile_finish $end_iteration --profile"
	if [ $profiler = 'ncu' ]
	then
        COMMAND="$COMMAND | awk '$AWK_CODE' >  $result_dir/batch_$batch_size.csv"
	fi

    echo $COMMAND
    eval sudo -E env PATH=$PATH $COMMAND
    sleep 5
done
popd > /dev/null

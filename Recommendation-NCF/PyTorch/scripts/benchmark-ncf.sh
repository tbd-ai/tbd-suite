#!/bin/bash
# source scripts/benchmark-ncf.sh # Train
# source scripts/benchmark-ncf.sh --profile # Profile for compute utilization
# source scripts/benchmark-ncf.sh --profile-fp32 # Profile for fp32 utilization

# Using a seed of 1
seed=1
skip_data=0
THRESHOLD=0.635

if [ $# -eq 0 ]
then
        echo "nvprof disabled"
        COMMAND="python3 ./source/ncf.py ml-20m -l 0.0005 -b 2048 --layers 256 256 128 64 -f 64 --seed $seed \
    	--threshold $THRESHOLD --processes 1 --workers 0"

elif [ "$1" == "--profile" ]
then
        echo "nvprof is profiling compute utilization"
        COMMAND="nvprof --profile-from-start off --export-profile nvprof_data/compute_utilization%p.nvvp --print-summary \
        python3 ./source/ncf.py ml-20m -l 0.0005 -b 2048 --layers 256 256 128 64 -f 64 --seed $seed \
        --threshold $THRESHOLD --processes 1 --workers 0 --profile"

elif [ "$1" == "--profile-fp32" ]
then
        echo "nvprof is profiling fp32 utilization"
        COMMAND="nvprof --profile-from-start off --metrics single_precision_fu_utilization --export-profile \
        nvprof_data/fp32_utilization%p.nvvp --print-summary python3 ./source/ncf.py ml-20m -l 0.0005 -b 2048 --layers \
        256 256 128 64 -f 64 --seed $seed --threshold $THRESHOLD --processes 1 --workers 0 --profile"

else
	echo "Invalid input argument. Valid ones are --profile --profile-fp32."
	return -1
fi


# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

echo "unzip ml-20m.zip"
if unzip ml-20m.zip
then
    if [ $skip_data -eq 0 ]
    then
        echo "Start processing ml-20m/ratings.csv"
        t0=$(date +%s)
        python3 ./source/convert.py ml-20m/ratings.csv ml-20m --negatives 999
        t1=$(date +%s)
        delta=$(( $t1 - $t0 ))
        echo "Finish processing ml-20m/ratings.csv in $delta seconds"
    else
        echo "Skipped data processing"
    fi

    echo "Start training"
    t0=$(date +%s)
    $COMMAND
    t1=$(date +%s)
	delta=$(( $t1 - $t0 ))
    echo "Finish training in $delta seconds"

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="recommendation"


	echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
else
	echo "Problem unzipping ml-20.zip"
	echo "Please run 'download_data.sh && verify_datset.sh' first"
fi







#!/bin/bash
# training configuration
profiler=${1:-'nvprof'}
start_iteration=${2:-1000}
profile_iteration=${3:-200}

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
export TF_CPP_MIN_LOG_LEVEL=3

AWK_CODE='{if(start==1) print($0); if (match($0, "==PROF== Disconnected from process")) start=1; }'

for batch_size in 1 2 4 8 16 32 64 96 128
do
    if [ $profiler = 'nvprof' ]
    then

        PREFIX="nvprof --profile-from-start off \
                --export-profile $result_dir/resnet50-tensorflow-${batch_size}-%p.nvvp -f"

        CMD=" $PREFIX python3 /app/source/resnet/src/resnet_ctl_imagenet_main.py --num_gpus=1 \
              --batch_size=$batch_size \
              --train_steps=1000 --steps_per_loop=$profile_iteration --log_steps=$profile_iteration\
              --dtype=fp32 --use_synthetic_data=false \
              --skip_eval \
              --data_dir=/app/data/TFRecords/ \
              --profile_step $start_iteration
              -model_dir=./model1 |& tee $result_dir/compute_util_batch_$batch_size.log"
        echo $CMD
        eval sudo -E env PATH=$PATH  $CMD
    else
        NSIGHT_PREFIX="ncu --csv --target-processes all --profile-from-start no --metrics sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,gpc__cycles_elapsed.max"

        CMD="$NSIGHT_PREFIX python3 /app/source/resnet/src/resnet_ctl_imagenet_main.py --num_gpus=1 \
              --batch_size=$batch_size \
              --train_steps=10000 --steps_per_loop=$profile_iteration --log_steps=$profile_iteration \
              --dtype=fp32 --use_synthetic_data=false \
              --skip_eval \
              --data_dir=/app/data/TFRecords/ \
              --profile_step $start_iteration
              -model_dir=./model1 | awk '$AWK_CODE' > tee $result_dir/core_util_batch_$batch_size.csv"
        echo $CMD
        eval sudo -E env PATH=$PATH  $CMD

    fi
done


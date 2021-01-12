#!/bin/bash

mkdir -p throughput_logs
echo $1
if [ ! -z "$1" ]
then
    end=$1
else
    end=1000
fi

for batch_size in 1 2 4 8 16 32 64 96 128
do

  CMD="python3 /app/source/resnet/src/resnet_ctl_imagenet_main.py --num_gpus=1 \
	  --batch_size=$batch_size \
	  --train_steps=$end --steps_per_loop=100 --log_steps=100 \
	  --dtype=fp32 --use_synthetic_data=false \
	  --skip_eval \
	  --data_dir=/app/data/TFRecords/ \
	  -model_dir=./model_temp | grep \"TimeHistory:\" | tee throughput_logs/throughput_$batch_size.log"
  echo $CMD
  eval $CMD

done



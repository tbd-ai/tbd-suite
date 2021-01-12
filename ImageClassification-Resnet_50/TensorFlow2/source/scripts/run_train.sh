#!/bin/sh

if [ $# -eq 0 ] 
then
    CMD="python3 /app/source/resnet/src/resnet_ctl_imagenet_main.py  \
      --batch_size=384 \
      --dtype=fp32 --use_synthetic_data=false \
      --data_dir=/app/data/TFRecords/ \
      --train_epochs=90 \
      --benchmark_log_dir ./logs \
      --steps_per_loop 100 \
      --all_reduce_alg nccl \
      --distribution_strategy multi_worker_mirrored \

	  -model_dir=./saved_resnet_imagenet_model |& tee training.log "
else
    CMD="python3 /app/source/resnet/src/resnet_ctl_imagenet_main.py  $*"
fi
echo $CMD
eval $CMD


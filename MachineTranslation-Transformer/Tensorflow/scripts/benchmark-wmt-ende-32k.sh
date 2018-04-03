#!/bin/sh

if [ "$1" = "" ]
then
  echo "Regular mode"
  python ../source/tensor2tensor/tensor2tensor/bin/t2t-trainer.py \
    --data_dir=../t2t_data \
    --problems=translate_ende_wmt32k \
    --model=transformer \
    --hparams_set=transformer_base_single_gpu \
    --output_dir=../t2t_train/base \
    --hparams='batch_size=2048'

elif [ "$1" = "--profile" ]
then
  echo "Profile mode"
  /usr/local/cuda/bin/nvprof --profile-from-start off --export-profile ../profile.nvvp -f --print-summary python ../source/tensor2tensor/tensor2tensor/bin/t2t-trainer.py \
    --data_dir=../t2t_data \
    --problems=translate_ende_wmt32k \
    --model=transformer \
    --hparams_set=transformer_base_single_gpu \
    --output_dir=../t2t_train/base \
    --train_steps=400 \
    --hparams='batch_size=2048'

elif [ "$1" = "--profile-fp32" ]
then
  echo "FP-32 Mode"
  /usr/local/cuda/bin/nvprof --profile-from-start off --export-profile ../profile_fp32.nvvp -f --print-summary --metrics single_precision_fu_utilization python ../source/tensor2tensor/tensor2tensor/bin/t2t-trainer.py \
    --data_dir=../t2t_data \
    --problems=translate_ende_wmt32k \
    --model=transformer \
    --hparams_set=transformer_base_single_gpu \
    --output_dir=..t2t_train/base \
    --train_steps=400 \
    --hparams='batch_size=2048'
else
    echo "Invalid input argument. Valid ones are --profile --profile-fp32."
    exit -1
fi


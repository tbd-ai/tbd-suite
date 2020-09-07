#!/bin/bash
counter=1
NUM_ITER=500

export MODEL_NAME=${counter}-first_generation
python3 bootstrap.py \
  --work_dir=estimator_working_dir \
  --export_path=outputs/models/$MODEL_NAME

while [ $counter -le ${NUM_ITER} ]
do
    export MODEL_NAME=${counter}-first_generation
    #echo $MODEL_NAME
    python3 selfplay.py \
        --load_file=outputs/models/$MODEL_NAME \
        --num_readouts 10 \
        --verbose 3 \
        --selfplay_dir=outputs/data/selfplay \
        --holdout_dir=outputs/data/holdout \
        --sgf_dir=outputs/sgf \
        --holdout_pct=0
    python3 selfplay.py \
        --load_file=outputs/models/$MODEL_NAME \
        --num_readouts 10 \
        --verbose 3 \
        --selfplay_dir=outputs/data/selfplay \
        --holdout_dir=outputs/data/holdout \
        --sgf_dir=outputs/sgf \
        --holdout_pct=0
    python3 selfplay.py \
        --load_file=outputs/models/$MODEL_NAME \
        --num_readouts 10 \
        --verbose 3 \
        --selfplay_dir=outputs/data/selfplay \
        --holdout_dir=outputs/data/holdout \
        --sgf_dir=outputs/sgf \
        --holdout_pct=100

    counter=$(( counter +1 ))
    #echo $counter
    python3 train.py \
        outputs/data/selfplay/* \
        --work_dir=estimator_working_dir \
        --export_path=outputs/models/${counter}-first_generation
    python3 validate.py \
         outputs/data/holdout \
        --work_dir=estimator_working_dir
done
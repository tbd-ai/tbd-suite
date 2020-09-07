#!/bin/bash
                                                                                                                                                                                                                         BS="32"

counter=1
GENERATION="fourth_generation"
export MODEL_NAME=${counter}-${GENERATION}
python3 bootstrap.py \
  --work_dir=estimator_working_dir \
  --export_path=outputs/models/$MODEL_NAME

while [ $counter -le 5 ]
do
    export MODEL_NAME=${counter}-${GENERATION}
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
    python3 train_mem_profile.py \
        outputs/data/selfplay/* \
        --work_dir=estimator_working_dir \
        --train_batch_size=${BS} \
        --export_path=outputs/models/${counter}-${GENERATION}
    python3 validate.py \
         outputs/data/holdout \
        --work_dir=estimator_working_dir
done

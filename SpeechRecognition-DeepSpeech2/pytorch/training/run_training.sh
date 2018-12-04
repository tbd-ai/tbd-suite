#!/usr/bin/env bash
# Script to train DeepSpeech 2 inference

RANDOM_SEED=1
TARGET_ACC=23

# ARG		CHOICE				HELP
# ${1} = integer				Looks for a model at new_training/deepspeech_${1}.pth.tar

MODELS="."
if [ "${1}" = "new" ]
then
	echo "Traning new model..."
	python training.py \
	    --checkpoint \
	    --model_path ${MODELS}/deepspeech_t$RANDOM_SEED.pth \
	    --seed $RANDOM_SEED --acc $TARGET_ACC
else
	echo "Traning from "${1}"..."
	python training.py \
	    --checkpoint \
	    --continue_from ${MODELS}/deepspeech_${1}.pth \
	    --model_path MODELS/deepspeech_t$RANDOM_SEED.pth \
	    --seed $RANDOM_SEED \
	    --acc $TARGET_ACC
fi

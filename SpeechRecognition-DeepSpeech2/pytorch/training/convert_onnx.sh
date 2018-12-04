#!/usr/bin/env bash
# Script to convert DeepSpeech2 models to ONNX fomat

RANDOM_SEED=1
TARGET_ACC=23

# ARG		CHOICE								HELP
# ${1} = {Integer >= 0}							Looks for a model at new_training/deepspeech_${1}.pth.tar

MODELS=../results/train/new_training
echo "Note: If there is an error, check to be sure you are running the .py3 Docker image!"
python3 ../source/convert_torch_to_onnx.py \
        --checkpoint \
        --continue_from MODELS/deepspeech_${1}.pth \
        --model_path MODELS/deepspeech_${1}.pth
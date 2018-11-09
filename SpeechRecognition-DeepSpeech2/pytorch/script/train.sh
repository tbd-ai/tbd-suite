# Script to train DeepSpeech 2 inference

RANDOM_SEED=1
TARGET_ACC=23

# ARG		CHOICE				HELP
# ${1} = integer				Looks for a model at new_training/deepspeech_${1}.pth.tar

MODELS=../results/train/new_training
if [ "${1}" -eq "new" ] then
	echo "Traning new model..."
	python train.py --checkpoint --model_path MODELS/deepspeech_t$RANDOM_SEED.pth.tar --seed $RANDOM_SEED --acc $TARGET_ACC
else
	echo "Traning from "${1}"..."
	python train.py --checkpoint --continue_from MODELS/deepspeech_${1}.pth.tar --model_path MODELS/deepspeech_t$RANDOM_SEED.pth.tar --seed $RANDOM_SEED --acc $TARGET_ACC
fi
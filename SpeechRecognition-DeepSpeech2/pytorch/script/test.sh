# Script to test and time DeepSpeech 2 implementation

RANDOM_SEED=1
TARGET_ACC=23

# ARG		CHOICE								HELP
# ${1} = {Integer >= 0}							Looks for a model at new_training/deepspeech_${1}.pth.tar
# ${2} = {libri, ov (coming soon)}				Chooses with dataset to use	for testing			
# ${3} = {Integer >= 0}							To use a homogenous dataset consisting of chiefly inputs of the same audio duration, set >=0. Else use -1 to ignore. 
# ${4} = float									The duration of the held input's audio clip in seconds used to normalize the timing stats. Set to 1 for no effect.
# ${5} = {Integer >= -1}						Max number of trials(batches) to infer. Causes the test script to infer the first n batches of the selected dataset. Set to -1 for no limit.

MODELS=../results/train/new_training
# By default we use GPU and batch size of 40
python test.py --cpu 0 --batch_size_val 40 --checkpoint --continue_from MODELS/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
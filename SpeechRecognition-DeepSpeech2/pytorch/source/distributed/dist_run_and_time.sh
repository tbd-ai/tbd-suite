# Script to train and time DeepSpeech 2 implementation

RANDOM_SEED=1
TARGET_ACC=23
NODE_RANK=${1}
WORLD=${2}
URL=${3}

python dist_train.py --checkpoint --model_path models/deepspeech_t$RANDOM_SEED.pth.tar --seed $RANDOM_SEED --acc $TARGET_ACC --node_rank $NODE_RANK --world_size $WORLD --dist_url $URL


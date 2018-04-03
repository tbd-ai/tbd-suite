DATASET_DIR=... # path to your TFRecords folder
TRAIN_DIR=./log

if [ "$1" = "" ]
then
        PREFIX=
        SUFFIX=

elif [ "$1" = "--profile" ]
then
        mkdir -p measurements
        PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
                --export-profile measurements/resnet50-tensorflow.nvvp -f --print-summary"
        SUFFIX=" --nvprof_on=True"

elif [ "$1" = "--profile-fp32" ]
then
        mkdir -p measurements
        PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
                --export-profile measurements/resnet50-tensorflow.nvvp -f \
                --metrics single_precision_fu_utilization"
        SUFFIX=" --nvprof_on=True"

else
        echo "Invalid input argument. Valid ones are --profile/--profile-fp32."; exit -1
fi

$PREFIX python ../source/train_image_classifier.py --train_dir=$TRAIN_DIR --dataset_dir=$DATASET_DIR \
	--model_name=resnet_v2_50 --optimizer=sgd --batch_size=32 \
	--learning_rate=0.1 --learning_rate_decay_factor=0.1 --num_epochs_per_decay=30 \
	--weight_decay=0.0001 $SUFFIX


DATASET_DIR=... # the path to your generated .rec file

if [ "$1" = "" ]
then
        PREFIX=
        SUFFIX=

elif [ "$1" = "--profile" ]
then
        mkdir -p measurements
        PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
                --export-profile measurements/inception-mxnet.nvvp -f --print-summary"
        SUFFIX=" --nvprof-on True"

elif [ "$1" = "--profile-fp32" ]
then
        mkdir -p measurements
        PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
                --export-profile measurements/inception-mxnet-fp32.nvvp -f \
                --metrics single_precision_fu_utilization"
        SUFFIX=" --nvprof-on True"

else
        echo "Invalid input argument. Valid ones are --profile/--profile-fp32."; exit -1
fi

$PREFIX python ../source/train_imagenet.py --gpus 0 --batch-size 32 --image-shape 3,299,299 --num-epochs 100 --network inception-v3 \
	--data-train $DATASET_DIR $SUFFIX

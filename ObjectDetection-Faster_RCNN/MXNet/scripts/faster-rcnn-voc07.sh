if [ "$1" = "" ]
then
        PREFIX=
        SUFFIX=

elif [ "$1" = "--profile" ]
then
        mkdir -p measurements
        PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
                --export-profile measurements/frcnn-mxnet.nvvp -f --print-summary"
        SUFFIX=" --nvprof-on True"

elif [ "$1" = "--profile-fp32" ]
then
        mkdir -p measurements
        PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
                --export-profile measurements/frcnn-mxnet.nvvp -f \
                --metrics single_precision_fu_utilization"
        SUFFIX=" --nvprof-on True"

else
        echo "Invalid input argument. Valid ones are --profile/--profile-fp32."; exit -1
fi

$PREFIX python ../source/train_end2end.py --network resnet $SUFFIX


protoc ../source/object_detection/protos/*.proto --python_out=../source
export PYTHONPATH=$PYTHONPATH:`pwd`/../source

if [ "$1" = "" ]
then
	PREFIX=
	SUFFIX=

elif [ "$1" = "--profile" ]
then
	mkdir -p measurements
	PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
		--export-profile measurements/faster-rcnn-tensorflow.nvvp -f --print-summary"
	SUFFIX=" --nvprof_on=True"

elif [ "$1" = "--profile-fp32" ]
then
	mkdir -p measurements
	PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
		--export-profile measurements/faster-rcnn-tensorflow.nvvp -f \
		--metrics single_precision_fu_utilization"
	SUFFIX=" --nvprof_on=True"

else 
	echo "Invalid input argument. Valid ones are --profile/--profile-fp32."; exit -1
fi

$PREFIX python2.7 ../source/object_detection/train.py --logtostderr --pipeline_config_path=../source/object_detection/samples/configs/faster_rcnn_resnet101_voc07.config --train_dir=tmp_frcnn $SUFFIX


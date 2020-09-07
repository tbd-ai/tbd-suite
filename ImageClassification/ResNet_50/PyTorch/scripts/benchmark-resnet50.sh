dataset=$1

if [ $# -eq 0 ]
then
	echo "No dataset path was provided. Please use: bash benchmark-resnet50.sh <path_to_dataset> <optional: --profile or --profile-fp32>"
	return -1

elif [ $# -eq 1 ]
then
        echo "nvprof disabled"
        COMMAND="python source/main.py -a resnet50 -b 32 $dataset"

elif [ "$2" == "--profile" ]
then
        echo "nvprof is profiling compute utilization"
        COMMAND="nvprof --profile-from-start off --export-profile nvprof_data/compute_utilization%p.nvvp --print-summary \
        python source/main.py -a resnet50 -b 32 $dataset"

elif [ "$2" == "--profile-fp32" ]
then
        echo "nvprof is profiling fp32 utilization"
        COMMAND="nvprof --profile-from-start off --metrics single_precision_fu_utilization --export-profile \
        nvprof_data/fp32_utilization%p.nvvp --print-summary python source/main.py -a resnet50 -b 32 $dataset"

else
	echo "Invalid input arguments. Please use: bash benchmark-resnet50.sh <path_to_dataset> <optional: --profile or --profile-fp32>"
	return -1
fi

$COMMAND

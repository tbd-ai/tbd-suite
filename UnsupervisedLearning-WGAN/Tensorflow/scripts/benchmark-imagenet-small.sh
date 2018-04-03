mkdir -p ../output
if [ "$1" == "" ]
then
	python ../source/gan_64x64.py
elif [ "$1" == "--profile" ] 
then
        /usr/local/cuda/bin/nvprof --profile-from-start off --export-profile ../profile.nvvp -f --print-summary python ../source/gan_64x64.py
elif [ "$1" == "--profile-fp32" ]
then
        /usr/local/cuda/bin/nvprof --profile-from-start off --metrics single_precision_fu_utilization --export-profile ../profile-fp32.nvvp -f --print-summary python ../source/gan_64x64.py
else
	echo "Invalid input argument. Valid ones are --profile --profile-fp32."
	exit -1
fi

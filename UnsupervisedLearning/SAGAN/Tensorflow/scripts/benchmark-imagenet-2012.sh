mkdir -p ./output
if [ "$1" == "" ]
then
	python ../source/train_imagenet.py --generator_type test --discriminator_type test --data_dir ../dataset/data/
elif [ "$1" == "--profile" ] 
then
        /usr/local/cuda/bin/nvprof --profile-from-start off --export-profile ./output/profile.nvvp -f --print-summary python ../source/train_imagenet.py --generator_type test --discriminator_type test --data_dir ../dataset/data/
elif [ "$1" == "--profile-fp32" ]
then
        /usr/local/cuda/bin/nvprof --profile-from-start off --metrics single_precision_fu_utilization --export-profile ./output/profile-fp32.nvvp -f --print-summary python ../source/train_imagenet.py --generator_type test --discriminator_type test --data_dir ../dataset/data/
else
	echo "Invalid input argument. Valid ones are --profile --profile-fp32."
	exit -1
fi

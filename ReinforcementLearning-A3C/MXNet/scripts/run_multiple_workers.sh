if [ "$1" == "" ]
then
	python ../source/launcher.py --gpus=0,1 -n 2 python ../source/a3c.py
elif [ "$1" == "--profile" ]
then
        nohup python2 launcher.py --gpus=0 -n 8 python2 a3c.py --profile-start=50000 --profile-stop=51000 --batch-size=64 &
        nohup /usr/local/cuda-8.0/bin/nvprof --export-profile profile-launcher-%p.nvvp -f --print-summary --profile-from-start off --profile-all-processes &
else
	echo "Invalid input argument. Valid ones are --profile."
	exit -1
fi

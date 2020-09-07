cd /source/machine_translation/nmt;

NSIGHT_PREFIX="/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --csv --target-processes all --profile-from-start no --metrics sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,gpc__cycles_elapsed.max"

for BATCH_SIZE in 32 64 `seq 128 128 1024`
do
	echo Testing batch size $BATCH_SIZE

	if [ $1 == "nsight" ]; then
		rm train_gnmt.py;
		ln -s train_gnmt.py.profile_nsight train_gnmt.py;

		$NSIGHT_PREFIX python3 train_gnmt.py --gpu 0 --epochs 1 --save_dir /scratch --batch_size $BATCH_SIZE |& tee -a /mnt/scripts/bs${BATCH_SIZE}.log
	elif [ $1 == "nvprof" ]; then
		rm train_gnmt.py;
		ln -s train_gnmt.py.profile_nvprof train_gnmt.py;

		NVPROF_PREFIX="nvprof --profile-from-start off --export-profile /scratch/measurements/profile_`date +%s`_bs${BATCH_SIZE}.nvvp"

		$NVPROF_PREFIX python3 train_gnmt.py --gpu 0 --epochs 1 --save_dir /scratch --batch_size $BATCH_SIZE 
	fi
done

#!/bin/bash
#FOLDER_OF_SECTION_FILE="/nsight_sections"
#SECTION_IDENTIFIER="PertBurstElapsed"
SECTION_IDENTIFIER="PertSustainedElapsed"


#NSIGHT_PREFIX="/usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli \
#--profile-from-start off \
#--csv \
#--section-folder ${FOLDER_OF_SECTION_FILE} \
#--section ${SECTION_IDENTIFIER} "

NSIGHT_PREFIX="/usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli \
--profile-from-start off \
--csv \
--metrics sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,\
gpc__cycles_elapsed.max \ "


mkdir /scratch/measurements;
cd /source/minigo;
export CUDA_VISIBLE_DEVICES=0;

NUM=1
NUM_GPUS=1

for BATCH_SIZE in 32 64 128
do
	echo Batch size $BATCH_SIZE

    APP="/usr/bin/python3 train_profiling.py outputs/data/selfplay/* \
    --work_dir=estimator_working_dir \
    --train_batch_size=${BATCH_SIZE} \
    --export_path=outputs/models/${NUM}-second_generation "

	if [ $1 == "nvprof" ]; then
		NVPROF_PREFIX="nvprof \
        --profile-from-start on \
        --profile-child-processes \
        --export-profile /scratch/measurements/profile_`date +%s`_bs${BATCH_SIZE}.nvvp "
		echo ${NVPROF_PREFIX}${APP}
        ${NVPROF_PREFIX}${APP}

	elif [ $1 == "nsight" ]; then
        NSIGHT_SUFFIX=" tee measurements/MiniGo_TF_${SECTION_IDENTIFIER}_num_gpus_${NUM_GPUS}_bs_${BATCH_SIZE}_nsight.csv"
        echo (${NSIGHT_PREFIX}${APP}) 2>&1 | ${NSIGHT_SUFFIX}  
        (${NSIGHT_PREFIX}${APP}) 2>&1 | ${NSIGHT_SUFFIX}  
	fi
done
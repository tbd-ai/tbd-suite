#!/bin/bash
#FOLDER_OF_SECTION_FILE="/nsight_sections"
#SECTION_IDENTIFIER="PertBurstElapsed"
SECTION_IDENTIFIER="PertSustainedElapsed"


NSIGHT_PREFIX="/opt/nvidia/nsight-compute/2020.1.0/nv-nsight-cu-cli \
--profile-from-start off \
--csv \
--section-folder ${FOLDER_OF_SECTION_FILE} \
--section ${SECTION_IDENTIFIER} "

# NSIGHT_PREFIX="/opt/nvidia/nsight-compute/2020.1.0/nv-nsight-cu-cli \
# --profile-from-start off \
# --csv \
# --metrics sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,\
# sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,\
# sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,\
# sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,\
# gpc__cycles_elapsed.max \ "


mkdir /scratch/measurements;
cd /source/deepspeech.pytorch;
export CUDA_VISIBLE_DEVICES=0;

NUM=1
NUM_GPUS=1
Train_Manifest_Path='/scratch/dataset/librispeech/libri_train_manifest.csv.new'
Val_Manifest_Path='/scratch/dataset/librispeech/libri_val_manifest.csv.new' 

for BATCH_SIZE in 4 8 16 24 32
do
	echo Batch size $BATCH_SIZE

    APP="python train_profiling.py \
    --train-manifest ${Train_Manifest_Path} \
    --val-manifest ${Val_Manifest_Path} \
    --opt-level O1 \
    --num-workers 8 \
    --batch-size ${BATCH_SIZE} \
    --epochs 1 \
    --log-dir log_nsight/ "

	if [ $1 == "nvprof" ]; then
		NVPROF_PREFIX="nvprof \
        --profile-from-start off \
        --profile-child-processes \
        --export-profile /scratch/measurements/profile_`date +%s`_bs${BATCH_SIZE}.nvvp "
		echo ${NVPROF_PREFIX}${APP}
        ${NVPROF_PREFIX}${APP}

	elif [ $1 == "nsight" ]; then
        NSIGHT_SUFFIX=" tee measurements/DS2_Pytorch_${SECTION_IDENTIFIER}_num_gpus_${NUM_GPUS}_bs_${BATCH_SIZE}_nsight.csv"
        echo (${NSIGHT_PREFIX}${APP}) 2>&1 | ${NSIGHT_SUFFIX}  
        (${NSIGHT_PREFIX}${APP}) 2>&1 | ${NSIGHT_SUFFIX}  
	fi
done
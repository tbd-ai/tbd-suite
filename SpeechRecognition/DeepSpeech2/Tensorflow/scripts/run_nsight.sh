#!/bin/bash
FOLDER_OF_SECTION_FILE="/workspace/Nsight_Sections"
#SECTION_IDENTIFIER="PertBurstElapsed"
SECTION_IDENTIFIER="PertSustainedElapsed"

#NSIGHT_PREFIX="/usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli \
#--profile-from-start off \
#--csv \
#--section-folder ${FOLDER_OF_SECTION_FILE} \
#--section ${SECTION_IDENTIFIER} "

NSIGHT_PREFIX="/usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli \
--csv \
--profile-from-start off \
--metrics sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,\
gpc__cycles_elapsed.max "

data_path="/SCRATCH/dataset/DS2_Librispeech_tf/"
final_train_file="${data_path}final_train_dataset.csv"
final_eval_file="${data_path}final_eval_dataset.csv" 

APP="/usr/local/bin/python deep_speech_profiling.py \
--train_data_dir=${final_train_file} \
--eval_data_dir=${final_eval_file} \
--num_gpus=1 \
--wer_threshold=0.23 \
--seed=1 "

NSIGHT_SUFFIX=" tee nsight_data/DS2_TF_${SECTION_IDENTIFIER}_num_gpus_1_bs_8_nsight.csv"
echo "(${NSIGHT_PREFIX}${APP}) 2>&1 | ${NSIGHT_SUFFIX}"
#(${NSIGHT_PREFIX}${APP})
(${NSIGHT_PREFIX}${APP}) 2>&1 | ${NSIGHT_SUFFIX}


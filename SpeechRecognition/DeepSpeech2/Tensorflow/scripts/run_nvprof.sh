#!/bin/bash
# Script to run deep speech model to achieve the MLPerf target (WER = 0.23)
# Step 1: download the LibriSpeech dataset.
#echo "Data downloading..."
#python data/download.py

data_path="/SCRATCH/dataset/DS2_Librispeech_tf/"
## After data downloading, the dataset directories are:
train_clean_100="${data_path}train-clean-100/LibriSpeech/train-clean-100.csv"
train_clean_360="${data_path}train-clean-360/LibriSpeech/train-clean-360.csv"
train_other_500="${data_path}train-other-500/LibriSpeech/train-other-500.csv"
dev_clean="${data_path}dev-clean/LibriSpeech/dev-clean.csv"
dev_other="${data_path}dev-other/LibriSpeech/dev-other.csv"
test_clean="${data_path}test-clean/LibriSpeech/test-clean.csv"
test_other="${data_path}test-other/LibriSpeech/test-other.csv"

# Step 2: generate train dataset and evaluation dataset
#icho "Data preprocessing..."
#train_file="${data_path}train_dataset.csv"
#eval_file="${data_path}eval_dataset.csv"

#head -1 $train_clean_100 > $train_file
#for filename in $train_clean_100 $train_clean_360 $train_other_500
#do
#    sed 1d $filename >> $train_file
#done

#head -1 $dev_clean > $eval_file
#for filename in $dev_clean $dev_other
#do
#    sed 1d $filename >> $eval_file
#done

# Step 3: filter out the audio files that exceed max time duration.
final_train_file="${data_path}final_train_dataset.csv"
final_eval_file="${data_path}final_eval_dataset.csv"

#MAX_AUDIO_LEN=27.0
#awk -v maxlen="$MAX_AUDIO_LEN" 'BEGIN{FS="\t";} NR==1{print $0} NR>1{cmd="soxi -D "$1""; cmd|getline x; if(x<=maxlen) {print $0}; close(cmd);}' $train_file > $final_train_file
#awk -v maxlen="$MAX_AUDIO_LEN" 'BEGIN{FS="\t";} NR==1{print $0} NR>1{cmd="soxi -D "$1""; cmd|getline x; if(x<=maxlen) {print $0}; close(cmd);}' $eval_file > $final_eval_file

# Step 4: run the training and evaluation loop in background, and save the running info to a log file
echo "Model training and evaluation..."
start=`date +%s`

log_file=log_`date +%Y-%m-%d`
nvprof \
  --profile-from-start on \
  --export-profile /workspace/deep_speech/nvprof_data/compute_data%p.nvvp \
  --profile-child-processes python deep_speech_profiling.py --train_data_dir=$final_train_file --eval_data_dir=$final_eval_file --num_gpus=-1 --wer_threshold=0.23 --seed=1 
#>$log_file 2>&1&

end=`date +%s`
runtime=$((end-start))
echo "Model training time is" $runtime "seconds."

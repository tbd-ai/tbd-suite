set -e

export LANG=C.UTF-8 
export LC_ALL=C.UTF-8

NSIGHT_PREFIX="/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --csv --target-processes all --profile-from-start yes --metrics sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,gpc__cycles_elapsed.max"

TMP_DIR=/dataset/t2t_datagen
DATA_DIR=/dataset/t2t_data
TRAIN_DIR=/scratch
TRANSLATIONS_DIR=/scripts

BEAM_SIZE=1

export CUDA_VISIBLE_DEVICES=0

t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=translate_ende_wmt32k

for BATCH_SIZE in 64 128 256 512 1024 2048 4096
do
	NVPROF_PREFIX="nvprof --profile-from-start off --export-profile /scratch/measurements/profile_`date +%s`_bs${BATCH_SIZE}.nvvp"

	rm -rf $TRAIN_DIR;
	echo Running batch size ${BATCH_SIZE}

	$NSIGHT_PREFIX t2t-trainer \
	  --data_dir=$DATA_DIR \
	  --problem=translate_ende_wmt32k \
	  --model=transformer \
	  --hparams_set=transformer_big_single_gpu \
	  --output_dir=$TRAIN_DIR \
	  --keep_checkpoint_max=1 \
	  --hp_batch_size $BATCH_SIZE | tee -a bs${BATCH_SIZE}.log
done

# t2t-trainer \
#   --data_dir=$DATA_DIR \
#   --problem=translate_ende_wmt32k \
#   --model=transformer \
#   --hparams_set=transformer_base_single_gpu \
#   --hparams="batch_size=$batch_size" \
#   --output_dir=$TRAIN_DIR \
#   --train_steps=$train_steps \
#   --worker-gpu=1 \
#   --schedule=train \
#   --keep_checkpoint_max=250 | tee -a ${TRAIN_DIR}/stdout.log

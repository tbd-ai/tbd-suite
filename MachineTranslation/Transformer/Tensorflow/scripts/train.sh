set -e

export LANG=C.UTF-8 
export LC_ALL=C.UTF-8

# export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

TMP_DIR=/dataset/t2t_datagen
DATA_DIR=/dataset/t2t_data
TRAIN_DIR=/scratch
TRANSLATIONS_DIR=/scripts

BEAM_SIZE=1

t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=translate_ende_wmt32k

t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=translate_ende_wmt32k \
  --model=transformer \
  --hparams_set=transformer_big_single_gpu \
  --output_dir=$TRAIN_DIR \
  --keep_checkpoint_max=250

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

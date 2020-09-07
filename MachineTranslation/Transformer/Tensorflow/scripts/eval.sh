set -e

export LANG=C.UTF-8 
export LC_ALL=C.UTF-8

TMP_DIR=/mnt/dataset/t2t_datagen
DATA_DIR=/mnt/dataset/t2t_data
TRAIN_DIR=/mnt/output
TRANSLATIONS_DIR=/mnt/scripts
LOG_DIR=/mnt/scripts/evaluation.log

SOURCE_TEST_TRANSLATE_DIR=${TMP_DIR}/dev/newstest2014-fren-src.en.sgm
REFERENCE_TEST_TRANSLATE_DIR=${TMP_DIR}/dev/newstest2014-fren-ref.en.sgm
BEAM_SIZE=1

batch_size=4096
train_steps=250000

BEAM_SIZE=4
ALPHA=0.6

#--decode_from_file=${TMP_DIR}/dev/newstest${year}-deen-src.en.sgm \
#--decode_from_file=${TMP_DIR}/dev/newstest2012-src.en.sgm \
for ITER in `seq 18000 1000 250000`
do
	echo Decoding checkpoint $ITER >> $LOG_DIR
	rm output.de;
	t2t-decoder \
	  --data_dir=$DATA_DIR \
	  --problem=translate_ende_wmt32k \
	  --model=transformer \
	  --checkpoint_path=/mnt/output/model.ckpt-${ITER} \
	  --hparams_set=transformer_big_single_gpu \
	  --output_dir=$TRAIN_DIR \
	  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
	  --decode_from_file=/mnt/dataset/wmt14/newstest2014.en \
	  --decode_to_file=output.de

	t2t-bleu --translation=output.de --reference=/mnt/dataset/wmt14/newstest2014.de >> $LOG_DIR
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

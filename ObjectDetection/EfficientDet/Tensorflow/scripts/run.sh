# begin training
DATASET_DIR=/dataset
cd /source/automl/efficientdet; 

rm det_model_fn.py;
ln -s det_model_fn.py.noprofile det_model_fn.py;

DATASET=coco
NUM_EPOCHS=50
LOG_FILE=/scratch/train_log

if [ $DATASET = "coco" ];
then
	python3 main.py --mode=train_and_eval \
	    --training_file_pattern=${DATASET_DIR}/tfrecord/train*.tfrecord \
	    --validation_file_pattern=${DATASET_DIR}/tfrecord/val*.tfrecord \
	    --val_json_file=${DATASET_DIR}/annotations/instances_val2017.json \
	    --model_name=efficientdet-d0 \
	    --model_dir=/tmp/efficientdet-d0-finetune \
	    --ckpt=efficientdet-d0  \
	    --train_batch_size=8 \
	    --eval_batch_size=8 --eval_samples=512 \
	    --num_examples_per_epoch=5717 --num_epochs=${NUM_EPOCHS}  \
	    --use_tpu=False |& tee -a $LOG_FILE
elif [ $DATASET = "pascal" ];
then
	python3 main.py --mode=train \
	    --training_file_pattern=${DATASET_DIR}/tfrecord/pascal*.tfrecord \
	    --validation_file_pattern=${DATASET_DIR}/tfrecord/pascal*.tfrecord \
	    --val_json_file=${DATASET_DIR}/tfrecord/json_pascal.json \
	    --model_name=efficientdet-d0 \
	    --model_dir=/tmp/efficientdet-d0-scratch  \
	    --backbone_ckpt=efficientnet-b0  \
	    --train_batch_size=8 \
	    --eval_batch_size=8 --eval_samples=512 \
	    --num_examples_per_epoch=5717 --num_epochs=${NUM_EPOCHS}  \
	    --hparams="num_classes=20,moving_average_decay=0" \
	    --use_tpu=False |& tee -a $LOG_FILE
fi

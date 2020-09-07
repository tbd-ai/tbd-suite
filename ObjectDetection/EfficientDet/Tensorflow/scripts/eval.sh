# begin training
SOURCES_ROOT=$(cd $(dirname $0) && pwd)/../source
DATASET_DIR="/dataset"
cd /source/automl/efficientdet; 

LOG_FILE=/scratch/eval_log

MODEL=efficientdet-d0
# CKPT_PATH=/tmp/efficientdet-d0-scratch
CKPT_PATH=/scratch/efficientdet-d0

python3 main.py --mode=eval \
    --model_name=${MODEL} \
    --model_dir=${CKPT_PATH} \
    --validation_file_pattern=${DATASET_DIR}/tfrecord/val* \
    --val_json_file=${DATASET_DIR}/annotations/instances_val2017.json  \
    --use_tpu=False |& tee -a $LOG_FILE

# --testdev_dir='/tmp/eval_output' \

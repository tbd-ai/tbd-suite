DATASET_DIR=/dataset
cd $DATASET_DIR

wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
rm train2017.zip val2017.zip annotations_trainval2017.zip

cd /mnt/source/automl/efficientdet/
python3 create_coco_tfrecord.py \
    --image_dir=${DATASET_DIR}/val2017 \
    --caption_annotations_file=${DATASET_DIR}/annotations/captions_val2017.json \
    --output_file_prefix=${DATASET_DIR}/tfrecord/val \
    --num_shards=32

python3 create_coco_tfrecord.py \
    --image_dir=${DATASET_DIR}/train2017 \
    --caption_annotations_file=${DATASET_DIR}/annotations/captions_train2017.json \
    --output_file_prefix=${DATASET_DIR}/tfrecord/train \
    --num_shards=32

python3 main.py --mode=eval  \
    --model_name=${MODEL}  --model_dir=${CKPT_PATH}  \
    --validation_file_pattern=${DATASET_DIR}/tfrecord/val*  \
    --val_json_file=${DATASET_DIR}/annotations/instances_val2017.json

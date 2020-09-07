PYTHONPATH=".:$PYTHONPATH" python3 dataset/create_coco_tfrecord.py \
  --image_dir=/mnt/dataset/train2017 \
  --caption_annotations_file=/mnt/dataset/annotations/captions_train2017.json \
  --output_file_prefix=/mnt/dataset/tfrecord/train \
  --num_shards=32

PYTHONPATH=".:$PYTHONPATH" python3 dataset/create_coco_tfrecord.py \
  --image_dir=/mnt/dataset/val2017 \
  --caption_annotations_file=/mnt/dataset/annotations/captions_val2017.json \
  --output_file_prefix=/mnt/dataset/tfrecord/val \
  --num_shards=32

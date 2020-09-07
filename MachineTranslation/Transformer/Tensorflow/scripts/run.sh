export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

t2t-trainer \
  --generate_data \
  --data_dir=/mnt/dataset \
  --output_dir=/mnt/dataset/mnist \
  --problem=image_mnist \
  --model=shake_shake \
  --hparams_set=shake_shake_quick \
  --train_steps=10000 \
  --eval_steps=100

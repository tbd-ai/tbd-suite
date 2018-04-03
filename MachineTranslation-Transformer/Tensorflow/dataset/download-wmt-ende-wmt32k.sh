mkdir -p ../t2t_data
python ../source/tensor2tensor/tensor2tensor/bin/t2t-datagen.py \
  --data_dir=../t2t_data \
  --problem=translate_ende_wmt32k
 

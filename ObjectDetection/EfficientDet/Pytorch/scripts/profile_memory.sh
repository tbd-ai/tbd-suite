cd /source/EfficientDet.Pytorch;
rm train.py;
ln -s train.py.noprofile train.py;
python3 train.py \
	--dataset COCO \
	--dataset_root /dataset \
	--save_folder /scratch \
	--network efficientdet-d0 \
	--batch_size 32 \
	--num_epoch 1 | tee -a /scratch/memprof.log

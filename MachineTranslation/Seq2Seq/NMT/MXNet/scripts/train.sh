cd /source/machine_translation/nmt;
rm train_gnmt.py;
ln -s train_gnmt.py.noprofile train_gnmt.py;
python3 train_gnmt.py --gpu 0 --save_dir /scratch/

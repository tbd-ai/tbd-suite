export PYTHONIOENCODING=UTF-8

# link train.py to the source file without profiling
rm /source/nmt/nmt/train.py
ln -s /source/nmt/nmt/train.py.noprofile /source/nmt/nmt/train.py

# begin training
cd /source/nmt;
python3 -m nmt.nmt \
    --src=vi --tgt=en \
    --batch_size 128 \
    --vocab_prefix=/dataset/vocab  \
    --train_prefix=/dataset/train \
    --dev_prefix=/dataset/tst2012  \
    --test_prefix=/dataset/tst2013 \
    --out_dir=/scratch \
    --hparams_path=/source/hparams/iwslt15.json


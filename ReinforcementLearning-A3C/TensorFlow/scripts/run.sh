rm -rf /tmp/a3c
python3 ../source/train.py --model_dir /tmp/a3c --env Breakout-v0 --t_max 5 --eval_every 300 --parallelism 8

INPUT_FILES='/home/danny/Documents/repos/tbd-suite/LanguageProcessing-BERT/TensorFlow/dataset/pretrain/tfrecord/training/sharded_training_0.tfrecord'
INPUT_DIR='/home/danny/Documents/repos/tbd-suite/LanguageProcessing-BERT/TensorFlow/dataset/pretrain/tfrecord/training/'
OUTPUT_FILES='/home/danny/Documents/repos/tbd-suite/LanguageProcessing-BERT/TensorFlow/dataset/pretrain/tfrecord/test/sharded_test_0.tfrecord'
python3 /home/danny/Documents/repos/tbd-suite/LanguageProcessing-BERT/TensorFlow/source/run_pretraining_custom.py \
    --input_dir=$INPUT_DIR \
    --output_dir='/home/danny/Documents/repos/tbd-suite/LanguageProcessing-BERT/TensorFlow/dataset/pretrain/results' \
    --bert_config_file='/home/danny/Documents/repos/tbd-suite/LanguageProcessing-BERT/TensorFlow/dataset/model/bert_config.json' \
    --do_train=True \
    --do_eval=True \
    --train_batch_size=8 \
    --eval_batch_size=4 \
    --max_seq_length 128 \
    --max_predictions_per_seq 20 \
    --num_train_steps 500 \
    --num_warmup_steps 50 \
    --save_checkpoints_steps 100 \
    --learning_rate 2e-5 \
    --optimizer_type='adam' 




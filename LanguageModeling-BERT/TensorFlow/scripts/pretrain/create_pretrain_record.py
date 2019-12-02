import subprocess


if __name__ == "__main__":
    max_seq_len = 128
    max_predictions_per_seq = 20
    masked_lm_prob = 0.15
    random_seed = 12345
    dupe_factor = 5
    output_file_prefix = 'sharded'

    def create_record_worker(filename_prefix, shard_id, output_format='tfrecord', split='training'):
        bert_preprocessing_command = 'python3 /home/danny/Documents/repos/tbd-suite/LanguageProcessing-BERT/TensorFlow/source/create_pretraining_data.py'
        bert_preprocessing_command += ' --input_file=' + '/home/danny/Documents/repos/tbd-suite/LanguageProcessing-BERT/TensorFlow/dataset/pretrain/sharded' + '/' + split + '/' + filename_prefix + '_' + str(shard_id) + '.txt'
        bert_preprocessing_command += ' --output_file=' + '/home/danny/Documents/repos/tbd-suite/LanguageProcessing-BERT/TensorFlow/dataset/pretrain/tfrecord' + '/' + split + '/' + filename_prefix + '_' + str(shard_id) + '.' + output_format
        bert_preprocessing_command += ' --vocab_file=' + '/home/danny/Documents/repos/tbd-suite/LanguageProcessing-BERT/TensorFlow/dataset/model/vocab.txt'
        #bert_preprocessing_command += ' --do_lower_case' if args.do_lower_case else ''
        bert_preprocessing_command += ' --max_seq_length=' + str(max_seq_len)
        bert_preprocessing_command += ' --max_predictions_per_seq=' + str(max_predictions_per_seq)
        bert_preprocessing_command += ' --masked_lm_prob=' + str(masked_lm_prob)
        bert_preprocessing_command += ' --random_seed=' + str(random_seed)
        bert_preprocessing_command += ' --dupe_factor=' + str(dupe_factor)
        bert_preprocessing_process = subprocess.Popen(bert_preprocessing_command, shell=True)

        last_process = bert_preprocessing_process

        # This could be better optimized (fine if all take equal time)
        # if shard_id % args.n_processes == 0 and shard_id > 0:
        #     bert_preprocessing_process.wait()

        return last_process

    for i in range(1):
        last_process = create_record_worker(output_file_prefix + '_training', i, 'tfrecord', 'training')

    last_process.wait()

    for i in range(1):
        last_process = create_record_worker(output_file_prefix + '_test', i, 'tfrecord', 'test')

    last_process.wait()

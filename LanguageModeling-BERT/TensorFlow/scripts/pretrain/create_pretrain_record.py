import subprocess
import sys

if __name__ == "__main__":
    max_seq_len = 128
    max_predictions_per_seq = 20
    masked_lm_prob = 0.15
    random_seed = 12345
    dupe_factor = 5
    output_file_prefix = 'sharded'
    create_pretrain_script = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    vocab_file = sys.argv[4]

    def create_record_worker(filename_prefix, shard_id, output_format='tfrecord', split='training'):
        bert_preprocessing_command = 'python3 ' + create_pretrain_script
        bert_preprocessing_command += ' --input_file=' + input_file + '/' + split + '/' + filename_prefix + '_' + str(shard_id) + '.txt'
        bert_preprocessing_command += ' --output_file=' + output_file + '/' + split + '/' + filename_prefix + '_' + str(shard_id) + '.' + output_format
        bert_preprocessing_command += ' --vocab_file=' + vocab_file
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

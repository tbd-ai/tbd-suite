cd ..
DIR=$(pwd)

echo "Extracting wiki data..."
python3 $DIR"/scripts/pretrain/WikiExtractor.py" -o $DIR"/dataset/pretrain/extracted/" $DIR"/dataset/pretrain/download/wiki_en.xml" 
echo "done!"

echo "Formatting wiki data..."
python3 $DIR"/scripts/pretrain/format_wiki_data.py" $DIR"/dataset/pretrain/extracted" $DIR"/dataset/pretrain/wiki/wikicorpus_en_one_article_per_line.txt"
echo "done!"

echo "Sharding wiki data..."
python3 $DIR"/scripts/pretrain/TextSharding.py" $DIR"/dataset/pretrain/wiki/wikicorpus_en_one_article_per_line.txt" $DIR"/dataset/pretrain/sharded/wiki"
echo "done!"

echo "Creating tf records..."
python3 $DIR"/scripts/pretrain/create_pretrain_record.py" $DIR"/source/create_pretraining_data.py" $DIR"/dataset/pretrain/sharded/wiki" $DIR"/dataset/pretrain/tfrecord" $DIR"/dataset/model/vocab.txt"
echo "done!"
cd ..
DIR=$(pwd)

echo "Extracting wiki data..."
python3 $DIR"/scripts/pretrain/WikiExtractor.py" -o $DIR"/dataset/pretrain/extracted/" $DIR"/dataset/pretrain/download/wiki_en.xml" 
echo "done!"

echo "Formatting wiki data..."
python3 $DIR"/scripts/pretrain/format_wiki_data.py" $DIR"/dataset/pretrain/extracted" $DIR"/dataset/pretrain/wiki/wikicorpus_en_one_article_per_line.txt"
echo "done!"

echo "Extracting wiki data..."
python3 ./pretrain/WikiExtractor.py -o ../dataset/pretrain/extracted/ ../dataset/pretrain/download/wiki_en.xml 
echo "done!"

echo "Formatting wiki data..."
python3 ./pretrain/format_wiki_data.py 
echo "done!"

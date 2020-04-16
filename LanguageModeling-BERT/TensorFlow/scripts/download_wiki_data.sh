cd ../dataset/

echo "setting up directories..."
mkdir model && mkdir pretrain
cd pretrain
mkdir download && mkdir extracted && mkdir formatted && mkdir sharded && mkdir tfrecord && mkdir results 
cd sharded && mkdir "test" && mkdir "training"

cd download 
echo "Downloading english wikipedia data..."
curl https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 --output wiki_en.xml.bz2 

echo "decompressing wikipedia data..."
bzip2 -d wiki_en.xml.bz2 

echo "done!"
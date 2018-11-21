# This file downloads the IWSLT15 English-Vietnamese dataset. 

NMT_ROOT=$(cd $(dirname $0) && pwd)/..

cd $NMT_ROOT/dataset; wget http://www.cs.toronto.edu/~bojian/Downloads/NMT/iwslt15-vi_en.tar.gz; \
	tar xvzf iwslt15-vi_en.tar.gz; rm -f iwslt15-vi_en.tar.gz

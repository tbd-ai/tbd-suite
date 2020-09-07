#! /bin/bash -e

# This file downloads the LibriSpeech ASR corpus from OpenSLR

DS2_ROOT=$(cd $(dirname $0) && pwd)/..

while [[ $# -gt 0 ]]
do
	case $1 in
		-d|--dataset)
			DATASET=$2
			shift
			;;
	esac
	shift
done

if [[ -z "$DATASET" ]]
then
	echo "Usage: ./download-librispeech.sh -d [dev-clean|dev-other|test-clean|test-other|train-clean-100|train-clean-360|traing-clean-500]"
	exit -1
fi

# Download dataset
rm -rf LibriSpeech
if [ ! -f $DATASET.tar.gz ]; then
	wget http://www.openslr.org/resources/12/$DATASET.tar.gz
fi
tar -xvzf $DATASET.tar.gz

# Convert to .wav
find . -iname "*.flac" | wc
for flacfile in `find . -iname "*.flac"`
do
    sox "${flacfile%.*}.flac" -e signed -b 16 -c 1 -r 16000 "${flacfile%.*}.wav"
done

# Construct JSON
python $DS2_ROOT/dataset/create_desc_json.py $DS2_ROOT/dataset/LibriSpeech/$DATASET corpus.json

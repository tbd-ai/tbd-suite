# Script to download Librispeech Dataset
# For best results allow downloader to use all dataset files.
# For the inference baseline, use the clean set.
# ARG		CHOICE				HELP
# ${1} = {all, clean, ...}		See the if statements for details.. selectively downloads data form librispeech

if [ "${1}" -eq "all" ] then
	echo "Downloading all..."
	python ../dataset/librispeech.py --target_dir ../dataset/LibriSpeech_dataset
	exit 0
fi
if [ "${1}" -eq "clean" ] then
	echo "Downloading clean only..."
	python ../dataset/librispeech.py --target_dir ../dataset/LibriSpeech_dataset --files_to_use train-clean-100.tar.gz,train-clean-360.tar.gz,dev-clean.tar.gz,test-clean.tar.gz
	exit 0
fi
if [ "${1}" -eq "clean_dev" ] then
	echo "Downloading clean dev only..."
	python ../dataset/librispeech.py --target_dir ../dataset/LibriSpeech_dataset --files_to_use train-clean-100.tar.gz,dev-clean.tar.gz,test-clean.tar.gz
	exit 0
fi
if [ "${1}" -eq "other" ] then
	echo "Downloading noisy, aka other only..."
	python ../dataset/librispeech.py --target_dir ../dataset/LibriSpeech_dataset --files_to_use train-other-500.tar.gz,dev-other.tar.gz,test-other.tar.gz
	exit 0
fi


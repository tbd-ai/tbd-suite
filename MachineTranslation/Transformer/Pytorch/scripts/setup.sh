
# install python3
apt update; apt install -y python3 python3-pip;

# install packages
pip3 install torch==1.5.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html;
pip3 install torchtext;
pip3 install spacy>=2.2.2;
pip3 install tqdm;
pip3 install dill;
pip3 install numpy;

# download dataset
python3 -m spacy download en
python3 -m spacy download de

# preprocess data
cd /mnt/source/attention-is-all-you-need-pytorch;
python3 preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl


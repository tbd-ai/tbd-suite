FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt update
RUN apt install -y python3 python3-pip wget git cuda-nsight-compute-10-1

# install python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install numba

RUN pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install pytablewriter \
	sacrebleu==1.2.10 \
	sacremoses==0.0.19 \
	git+git://github.com/rsennrich/subword-nmt.git@48ba99e657591c329e0003f0c6e32e493fa959ef \
	git+git://github.com/NVIDIA/apex.git#egg=apex

WORKDIR /scripts

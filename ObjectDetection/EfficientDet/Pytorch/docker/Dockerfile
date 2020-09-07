FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt update
RUN apt install -y python3 python3-pip wget git cuda-nsight-compute-10-1
RUN apt install -y curl unzip

# install torch
RUN pip3 install --user torch==1.5+cu101 torchvision==0.6+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install other dependencies
RUN pip3 install numba

WORKDIR /scripts

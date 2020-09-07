FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt update
RUN apt install -y python3 python3-pip wget git cuda-nsight-compute-10-1

# install python dependencies
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir numba

RUN pip3 install --no-cache-dir torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir OpenNMT-py

WORKDIR /scripts

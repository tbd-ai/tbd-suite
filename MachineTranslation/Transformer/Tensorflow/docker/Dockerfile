FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt update
RUN apt install -y python3 python3-pip wget cuda-nsight-compute-10-1

# install python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install numba

RUN wget -nc https://www.cs.toronto.edu/~ybgao/tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl
RUN pip3 install tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl
RUN rm tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl

WORKDIR /scripts

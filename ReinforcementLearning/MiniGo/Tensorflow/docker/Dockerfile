ARG UBUNTU_VERSION=18.04
ARG CUDA=10.0
FROM nvidia/cuda:${CUDA}-cudnn7-devel-ubuntu${UBUNTU_VERSION}

#ENV LC_CTYPE=en_US.UTF-8
#ENV LANG=en_US.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
 locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# Install general dependencies
RUN apt-get update && apt-get install -y python3 python3-pip rsync git wget pkg-config zip g++ zlib1g-dev unzip cuda-nsight-compute-10-0

# install python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install numba
RUN pip install setuptools --upgrade
RUN pip install six --upgrade
RUN pip3 install "tensorflow-gpu==1.15.0"


ENV SRC_DIR ${HOME}/source/minigo
RUN pip install -r $SRC_DIR/requirements.txt

WORKDIR /scripts

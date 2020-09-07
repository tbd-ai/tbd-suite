FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 

# Install Pip and python
RUN apt-get update \
    && apt-get install -y python3-pip python3-dev \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip

# Install Abseil python common libraries
RUN pip install absl-py \
    && pip3 install absl-py

# Install libSox
RUN apt-get install sox \
    && apt-get install libsox-fmt-mp3

# Install tensorflow
RUN pip install setuptools --upgrade \
    && pip install "tensorflow==1.15"

# Install vim
RUN apt-get install vim

# Install subversion library
RUN apt-get install subversion

# Insatll the HTTP library for python
RUN pip install requests

RUN cd /workspace \
  && cd source/deep_speech/ \
  && pip install -r requirements.txt

WORKDIR /scripts

FROM seannaren/deepspeech.pytorch:latest

# Install Pip and python
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
# Install PyTorch
RUN pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Install this fork for Warp-CTC binding
RUN cd /workspace \
  && git clone https://github.com/SeanNaren/warp-ctc.git \
  && cd warp-ctc; mkdir build; cd build; cmake ..; make \
  && export CUDA_HOME="/usr/local/cuda" \
  && cd ../pytorch_binding && python setup.py install

# Install NVIDIA apex
RUN cd /workspace \
  && git clone --recursive https://github.com/NVIDIA/apex.git \
  && cd apex && pip install .

# Finally clone this repo and run this within the repo
RUN cd /workspace \
  && cd deepspeech.pytorch/ \
  && pip install -r requirements.txt

# Multi-GPU/Multi-node training, you'll need etcd.
RUN apt-get install etcd

# install other dependencies
RUN pip3 install numba

WORKDIR /scripts
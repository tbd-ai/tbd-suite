FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
RUN apt-get update && apt-get install -y \
    curl \
    sudo \
    git \
    vim \
 && rm -rf /var/lib/apt/lists/*
# Create a working directory
RUN mkdir /app
WORKDIR /app


ARG host_uid
# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user --uid $host_uid\
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user
# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-3.19.0-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.6 \
 && conda clean -ya

RUN curl -sLo ./tensorflow_gpu-2.2.0-cp36-cp36m-manylinux2010_x86_64.whl https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-2.2.0-cp36-cp36m-manylinux2010_x86_64.whl \
        && pip install --upgrade pip setuptools  \
        && pip install --user --upgrade tensorflow-model-optimization numba  \
        && pip install tensorflow_gpu-2.2.0-cp36-cp36m-manylinux2010_x86_64.whl --ignore-installed six \
        && rm tensorflow_gpu-2.2.0-cp36-cp36m-manylinux2010_x86_64.whl

COPY ./nsight-compute-linux-2020.1.0.33-28294165.run .
USER root
RUN  bash ./nsight-compute-linux-2020.1.0.33-28294165.run --quiet -- -noprompt && \
        rm nsight-compute-linux-2020.1.0.33-28294165.run

USER user
RUN echo "source activate " > ~/.bashrc
ENV PATH /opt/conda/envs/tbd-resnet-torch/bin:/usr/local/NVIDIA-Nsight-Compute-2020.1:$PATH

USER user

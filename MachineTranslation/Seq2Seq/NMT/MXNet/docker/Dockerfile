FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt update
RUN apt install -y python3 python3-pip wget git cuda-nsight-compute-10-1

# install python dependencies
RUN pip3 install --upgrade pip

RUN pip3 install asn1crypto==0.24.0 \
	certifi==2020.4.5.1 \
	chardet==3.0.4 \
	cryptography==2.1.4 \
	Cython==0.29.17 \
	gluonnlp==0.9.1 \
	graphviz==0.8.4 \
	idna==2.6 \
	keyring==10.6.0 \
	keyrings.alt==3.0 \
	mxnet-cu101>=1.6.0 \
	numpy==1.18.4 \
	packaging==20.3 \
	pycrypto==2.6.1 \
	pygobject==3.26.1 \
	pyparsing==2.4.7 \
	pyxdg==0.25 \
	requests==2.23.0 \
	SecretStorage==2.3.1 \
	six==1.11.0 \
	urllib3==1.25.9 \
	numba

WORKDIR /scripts

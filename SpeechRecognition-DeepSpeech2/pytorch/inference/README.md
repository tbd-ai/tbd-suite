# Inference

## Basic Instructions

Machine requirements: Ubuntu 16.04, 15 GB disk
Choose VM instance(s):
 
- Azure F8s_v2 or better,  NC family

Clone this repository. Download the "deepspeech_20.pth.tar" model (from https://drive.google.com/drive/u/1/folders/1OioL2tqOsVWNW0j_I6J7gZxneFBd2gsB) and place it under the inference folder.
Cd to the inference directory then run (if CPU only):

	sh setup.sh

If you want GPU support run instead:

	sh setup.sh cuda

The above set up will

- Download python and necessary libraries
- Nividia driver and cuda if user asked for GPU support
- LibriSpeech clean test dataset 

and issue two commands in the final two lines of the execution for you to run next. They should be:
	
	newgrp docker
	sh ../docker/run_{cuda_}dev.sh

There will be the {cuda_} portion if you used GPU and nothing if you used the default CPU only setup. 
The run_dev script brings you inside the docker contianer where you can run the inference from by running:

	cd <path/to/this/inference/folder>
	sh run_inference.sh
	
The default settings will run inference with

- Batchsize 1
- LibriSpeech Test Clean
- CPU only

## Advanced Instructions

For the advance user we have provided details that underlies the steps taken by the setup.sh script.
Machine requirements: Ubuntu 16.04, 15 GB disk, roughly:

- 1 GB for dataset
- 5 GB for docker image
- 9 GB overhead and model weights
- Graphics card is optional (choosing one with more memory is better)

Software dependencies:

- sox
- libsox-fmt-mp3
- Python 2.7
- Python sox, wget
- modified wrap-ctc (from https://github.com/ahsueh1996/warp-ctc.git)
- Python h5py
- Python hickle
- Python tqdm
- Python pytorch 0.4.0 (from http://download.pytorch.org/whl/cu80/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl)
- Python cffi
- Python python-Levenshtein

GPU Software dependencies:

- Correct Nvidia driver
- Cuda driver
- Cuda 9 

You should use docker to ensure that you are using the same environment but you can build a conda environment, but just be cautious with the dependencies. Check into the setup.sh and ../docker/Dockerfile.gpu for exact details.

### Building Docker Image (Recommended)

Using docker is the simplist way to get all of the dependencies listed above. First we need to get docker.
For CPU only this is done with:

	cd ../docker
	install_docker.sh
	# ---- or equivalently ----
	sudo apt install docker.io

For GPU support run:

	cd ../docker
	sudo add-apt-repository ppa:graphics-drivers/ppa
	sudo apt-get update
	sudo apt-get install nvidia
	sudo apt-get install cuda-drivers
	install_cuda_docker.sh

To run docker, we need to add the user to the `docker` user group, run:

	sudo usermod -a -G docker $USER
	newgrp docker
	
We now have the correct docker support to build the images, run:

	cd ../docker
	# ---- CPU only ----
	sh build_docker.sh
	# ---- or if GPU support ----
	sh build_cuda_docker.sh

which will build a docker image based on `Dockerfile.gpu`. To see if the image has been successfully built, you should see your new image listed by running:

	docker images

To enter the contianer, simply run:

	cd ../docker
	# ---- CPU only ----
	sh run_dev.sh
	# ---- or if GPU support ----
	sh run_cuda_dev.sh

### Dataset

We use ["OpenSLR LibriSpeech Corpus"](http://www.openslr.org/12/) dataset, which provides over 1000 hours of speech data in the form of raw audio. It includes:
	
- train-clean-100.tar.gz
- train-clean-360.tar.gz
- train-other-500.tar.gz
- dev-clean.tar.gz
- dev-other.tar.gz
- test-clean.tar.gz
- test-other.tar.gz

When downloading the dataset, you will need the `sox, wget` and `libsox-fmt-mp3` dependencies.
You may choose to download the dataset after entering the docker container but it is fine to download without docker also.
Only do the following if you are outside your docker container:

	sudo apt-get install python-pip
	pip install sox wget
	sudo apt-get install sox libsox-fmt-mp3
	
For inference, we use clean dataset only. Specifically only the `test-clean.tar.gz` file will be used. Run:

	sh download_dataset.sh clean_test

which takes around 1.5 mins and uses 1 GB of disk space.
The download script will do some preprocessing and audio file extractions. Here are some things to note:
	
  - Data preprocessing:
    - The audio files are sampled at 16kHz.
    - All inputs are also pre-processed into a spectrogram of the first 13 mel-frequency cepstral coefficients (MFCCs).
	
  - Training and test data separation:
    - After running the `download_dataset` script, the `LibriSpeech_dataset` directory will have subdirectories for training set, validation set, and test set.
    - Each contains both clean and noisy speech samples.

  - Data order:
    - Audio samples are sorted by length.

### Running Inference

Download the "deepspeech_20.pth.tar" model (from https://drive.google.com/drive/u/1/folders/1OioL2tqOsVWNW0j_I6J7gZxneFBd2gsB) and place it under the inference folder.
Make sure you are inside your docker contianer. One way to check is to try `git` and seeing that it is not installed or simply exiting your session and running:

	cd ../docker
	# ---- CPU only ----
	sh run_dev.sh
	# ---- or if GPU support ----
	sh run_cuda_dev.sh
	
Then:

	cd <path/to/this/inference/folder>
	sh run_inference.sh

You may edit `run_inference.sh` to change the batchsizes of the inference.

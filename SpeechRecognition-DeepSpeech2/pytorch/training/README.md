# Training

## Environment set up
Environment requirement: Ubuntu 16.04 with sox installed, 200GB disk  

We reconmend building docker image to be consistent with our development environment. And we will provide docker set up scripts here. Alternatively, you can build a conda environment, but just be cautious with the dependencies.  

### Build docker image

Before proceeding, please make sure docker has been installed. You can install docker using the following script:
```
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt update
sudo apt install docker-ce -y
```

Then cd to docker folder, run:

```
sh build_cuda_docker.sh
```
which will build a docker image based on `Dockerfile.gpu` which lists library dependencies for running both training & inference. To see whether an image has been successfully built, you should see your newly built image after running:
```
docker images
```
	
## Dataset

We use ["OpenSLR LibriSpeech Corpus"](http://www.openslr.org/12/) dataset, which provides over 1000 hours of speech data in the form of raw audio. The dataset itself is over 100GB, and intermediate files during download, processing, and verification require 220GB of free disk space to complete.

#### The following assumes working directory dataset/

### Download

For training only, you only need to download training and validation LibriSpeech dataset, run

```
sh download_dataset.sh train
```
It will download and process the full training LibriSpeech dataset, and dump it into `./LibriSpeech_dataset`, which takes up to 4 hours. Make sure you have 200 GB of free disk. 

Some notes:	
  - Data preprocessing:
    - The audio files are sampled at 16kHz.
    - All inputs are also pre-processed into a spectrogram of the first 13 mel-frequency cepstral coefficients (MFCCs).
	
  - Training and test data separation:
    - After running the `download_dataset` script, the `LibriSpeech_dataset` directory will have subdirectories for training set, validation set, and test set.
    - Each contains both clean and noisy speech samples.
	
  - Data order:
    - Audio samples are sorted by length.

## Run training
After successfully building the docker image, you can proceed to launch the docker instance (making path modifications as necessary), and start training & testing.

To launch the docker instance, cd to docker/ and run:
```
sh run_dev.sh
```

NOTE: remember to modify paths in `docker/run-dev.sh` as appropriate (e.g., replace line 3 with the base path for the repo `~/mlperf/speech_recognition` or similar).

You should be in the docker container after running the above command. To start training, navigate to pytorch/ and run:

```
sh script/run_training.sh new | tee my_training_log.out
```

which will start training a new model and recording all std output to `my_training_log.out`. It will run until the specified target accuracy is achieved or 10 full epochs have elapsed, whichever is sooner. The maximum number of epochs, along with other network parameters, can be viewed and modified in `model/params.py`. 

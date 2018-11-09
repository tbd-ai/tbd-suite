# 1. About
This version of DeepSpeech2 is adapted from ML-perf for the purpose of providing you a useable model for speech recognition.
You can find the ML-perf repository: https://github.com/mlperf/training/commit/9c6b8b3ed27c2c8bc1e9f697b9d7c0ed75a72efb
You can find our pytorch and ONNX models: https://drive.google.com/drive/u/1/folders/1K3aIu2qm1R2h55C4-qgGlYOOID6sohru

# 2. DISCLAIMER: Model

The model size was updated to be more representative of speech recognition networks deployed in industry.
The trained models are the vanilla models (good for forming a standardized benchmark) however if you would like a higher accuracy, you will have to opitimize the hyperparameters or the data-preprocessing further.

# 3. Directions
### Workflow
1. Install nvidia, cuda and audio drivers and support
2. Build a docker image for your machine
3. Download the dataset
4. Run the docker image
5. Run training
6. Run testing

### Steps 1-2: Configure machine
Suggested environment : Ubuntu 16.04, 8 CPUs, one P100, 300GB disk

Assume sufficiently recent NVIDIA driver is installed. Check with
	nvidia-smi

Here is a suggestion for how to get a driver (but you should really _check out which driver version is required for your hardware_).
	sudo add-apt-repository ppa:graphics-drivers/ppa
	sudo apt-get update
	sudo apt-get install cuda-drivers

You may need Cuda 9.0:

    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.de
    sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install cuda-libraries-9-0

You may need to install docker before using our script:

    sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo apt-key fingerprint 0EBFCD88
    sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
       $(lsb_release -cs) \
       stable"
    sudo apt update
    sudo apt install docker-ce -y

Use our script to install nvidia-docker (Note, echo $USER to see that your username is set as a environment variable):
	sh script/docker/install_cuda_docker.sh
Crucially, add your username to the docker usergroup to avoid having to sudo run the container (causes problems later).
	sudo usermod -a -G docker $USER
	newgrp docker

Use our script to build your docker image:
	sh script/docker/build-docker.sh
There are 14 sections in this script. This script is installing pytorch, torchaudio, pip, and various other dependencies to the image. To check for completion:
	docker images
You should see
	ds2-cuda9cudnn
	
### Step 3: Download and verify data
The `download_dataset` script will use the python data utils defined in the `data` directory to download and process the full LibriSpeech dataset into `./LibriSpeech_dataset`.  This takes up to 6 hours.
The `verify_dataset` script will build a single tarball of the dataset, checksum it, compare against a reference checksum, and report whether they match.  This takes up to an hour.
The dataset itself is over 100GB, and intermediate files during download, processing, and verification require 220GB of free disk space to complete.
Here is a template idea that uses a extra disk to hold the dataset:
	sudo mkdir /scratch
	sudo mount /dev/sdb1 /scratch
	sudo chown -R $USER:docker /scratch
Clone our repo onto /scratch and proceed to downloading the dataset in that directory. Then use the link:
	ln -s /scratch/you/directory/libri_train_manifest.csv /home/$USER/your/directory/

We suggest using tmux to encapsulate the actual download which will take a while.
	tmux new -s tbd_ds2
	tmux attach -t tbd_ds2
    sh script/download_dataset.sh all
	
We provide several options in the download script: all, clean, clean_small, other. These refer to the actual samples that the training and inference will consider.

Note that the example verification script works only with "all" because only it's reference checksum is provided.
You should also run verification inside the docker container (see next step).

Publication/Attribution:
	["OpenSLR LibriSpeech Corpus"](http://www.openslr.org/12/) provides over 1000 hours of speech data in the form of raw audio.
Data preprocessing:
	The audio files are sampled at 16kHz.
	All inputs are also pre-processed into a spectrogram of the first 13 mel-frequency cepstral coefficients (MFCCs).
Training and test data separation:
	fter running the `download_dataset` script, the `LibriSpeech_dataset` directory will have subdirectories for training set, validation set, and test set.
	Each contains both clean and noisy speech samples.
Data order:
	Audio samples are sorted by length.

### Step 4: Run docker container
Now we can spin the docker image up into an actual container. We suggest using tmux as well.
	tmux new -s tbd_ds2
	tmux attach -t tbd_ds2
	sh script/docker/run-dev.sh

This script will add the /scratch folder and the /home/$USER/your/directory/ as volumes so you can pass files to and from the container and your parent OS.

### Steps to run and time
For each framework, there is a provided docker file and `run_and_time.sh` script.
To run the benchmark, (1) build the docker image, if you haven't already, (2) launch the docker instance (making path modifications as necessary), and (3) run and time the `run_and_time` script, optionally piping output to a log file.
For example, for the pytorch framework:

    cd pytorch
    cd docker
    sh build-docker.sh
    sh run-dev.sh
    time sh run_and_time.sh | tee speech_ds2.out

NOTE: remember to modify paths in `docker/run-dev.sh` as appropriate (e.g., replace line 3 with the base path for the repo `~/mlperf/reference/speech_recognition` or similar).

The model will run until the specified target accuracy is achieved or 10 full epochs have elapsed, whichever is sooner. The maximum number of epochs, along with other network parameters, can be viewed and modified in `pytorch/params.py`.

### Step 5-6: Training and testing
Simply run from inside the docker container:
	sh script/train.sh new | tee my_training_log.out
	sh script/test.sh 20 libri -1 1 -1 | tee my_inference_result.out

# 5. Model
### Publication/Attribution
This is an implementation of [DeepSpeech2](https://arxiv.org/pdf/1512.02595.pdf) adapted from [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch).
### List of layers
Summary: Sampled Audio Spectrograms -> 2 CNN layers -> 5 Bi-Directional GRU layers -> FC classifier layer -> output text

Details:

  (module): DeepSpeech (

    (conv): Sequential (

      (0): Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2))

      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)

      (2): Hardtanh (min_val=0, max_val=20, inplace)

      (3): Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1))

      (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)

      (5): Hardtanh (min_val=0, max_val=20, inplace)

    )
    (rnns): Sequential (

      (0): BatchRNN (


        (rnn): GRU(672, 2560)
      )

      (1): BatchRNN (

        (rnn): GRU(2560, 2560)

      )

      (2): BatchRNN (

        (rnn): GRU(2560, 2560)

      )

    )

    (fc): Sequential (

      (0): SequenceWise (

      Sequential (

        (0): BatchNorm1d(2560, eps=1e-05, momentum=0.1, affine=True)

        (1): Linear (2560 -> 29)

      ))

    )

    (inference_log_softmax): InferenceBatchLogSoftmax (

    )

  )

)

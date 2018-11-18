# 1. About
This version of DeepSpeech2 is adapted from ML-perf for the purpose of providing you a useable model for speech recognition.
You can find the ML-perf repository: https://github.com/mlperf/training/commit/9c6b8b3ed27c2c8bc1e9f697b9d7c0ed75a72efb
You can find our pytorch and ONNX models: https://drive.google.com/drive/u/1/folders/1K3aIu2qm1R2h55C4-qgGlYOOID6sohru

# 2. DISCLAIMER: Model

The model size was updated to be more representative of speech recognition networks deployed in industry.
The trained models are the vanilla models (good for forming a standardized benchmark) however if you would like a higher accuracy, you will have to opitimize the hyperparameters or the data-preprocessing further.

# 3. Directions
The following is the standard workflow for those who intend to run both training and inference. For people who merely just want to run training or inference, please refer to README under the corresponding folder.

## Workflow
1. Environment set up
2. Download & preprocess dataset
3. Run training
4. Run inference

## Env set up
Environment requirement: Ubuntu 16.04 with sox installed, 300GB disk  

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
Run
```
sh download_dataset.sh all
```
This will download and process the full LibriSpeech dataset, and dump it into `./LibriSpeech_dataset`, which takes up to 6 hours. Make sure you have 300 GB of free disk.  

For purpose of testing & inference, we use clean dataset only, e.g.
```
sh download_dataset.sh clean_dev
```
which takes around 25 mins and uses 15 GB of disk space.  

Some notes:	
  - Data preprocessing:
    - The audio files are sampled at 16kHz.
    - All inputs are also pre-processed into a spectrogram of the first 13 mel-frequency cepstral coefficients (MFCCs).
	
  - Training and test data separation:
    - After running the `download_dataset` script, the `LibriSpeech_dataset` directory will have subdirectories for training set, validation set, and test set.
    - Each contains both clean and noisy speech samples.
	
  - Data order:
    - Audio samples are sorted by length.

### Run training and testing
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

To do testing, inside the docker container run the following after training completed:

```
sh inference/run_inference.sh 10 libri -1 1 -1 | tee my_inference_result.out
```

# 4. Model
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

# 5. Quality
Best WER 21.2 on Librispeech Test Clean.
Navigate to results/ for more details.

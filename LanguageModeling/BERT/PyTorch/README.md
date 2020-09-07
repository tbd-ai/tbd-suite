# LanguageModeling-Bert-PyTorch

## Description
This is a benchmarking suite for the BERT model under PyTorch.
The benchmarking source code is adopted from NVIDIA's Deep-learning Example: [github link](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)


The benchmark is performed using the following software environments
* Ubuntu version: 18.04
* CUDA version:   10.1
* CUDNN version:  7.6
* PyTorch vresion: 1.4

## Hardware Requirements

In order to run the benchmark suite, your machines need to have NVIDIA GPUs with CUDA installed. The CUDA diver version on your machine needs to be >= 418.39 as according to the  [CUDA compatilibility page](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility).

## Setup
Please refer to the following steps to set up the environment to perform benchmarks.

### Installing Docker
[Docker](https://docs.docker.com/get-started/overview/) is an open-platform for developing, shipping, and running applications in a unified, isolated, and packaged environment called containers. This benchmark suites runs within a docker container with a docker version = 19.03. Please refer to the instructions in https://docs.docker.com/engine/install/#server to install docker on your own host device. You will also need to install the NVIDIA runtime for docker to enabnle GPUs inside the container, the intructions are available at https://github.com/NVIDIA/nvidia-container-runtime#installation.

### Building Docker Image
If this is your first time running the benchmark on the host, please prepare the environment by running the following command. This will install a docker image with the required software versions and packages. 

Before building the image, you need to download the installer of NVIDIA® Nsight™ Compute as it is not included in the repo due to size constraints. You can download it from https://developer.nvidia.com/rdp/assets/nsight-compute-2020_1_1_8-Linux-installer, note that you will need to sign up for NVIDIA Developer Membership (free) in order to download the installer. Once it is doanloeded, copy it under the "docker" dicrectory.

```sh
cp <Location of Nsight Compute Installer> ./docker/
bash ./build_image.sh
```
This will take approximately 5 minutes, and you should see the following message when it finishes
>>Successfully built xxxxxxxxxx
>>Successfully tagged tbd-bert-pytorch:latest

### Running the Container
Next, start a docker container by running
```sh
bash ./start_container.sh <dataset dir>
```
This will prompt a shell in the container environment. You will need to provide a dataset directory to ./run_container.sh with the format specified in section [Preparing the dataset](###preparing-the-dataset). If you do not currently have the dataset, please provide an empty directory to be mounted to the container and populated from within the container in the following sections.

Make sure you are inside of a the docker container by observing the command prompt:
>>(bert_pytorchh) user@xxxxxxxxxxx:


### Preparing the Dataset

We use Wikipedia and BookCorpus as our pretraining dataset. The downloader are included under the "source" direcotry. The datasets are then further sharded and converted into hdf5 format. To download the dataset, run the following command:
```sh
 bash /workspace/bert/data/create_datasets_from_start.sh 
 ```
This will use Wikiextractor and BookCorpus to download raw text, clean and format the text, and finally shard and convert them into hdf5 format. The download can take up to 1 day, and after downloading you sould see two folders created under /workspace/bert/data, _"hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5"_
and
_"hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5"_
Which are used for phase 1 and phase 2 of the pre-training process.

It will also download the SQuAD for fine-tuning the question-answering task.


## Steps to Train and Profile
Note that in this benchmark we have chosen to use the default hyper-parameters of the model, as specified in  /workspace/bert/scripts/run_pretraining.sh

### Running End to End Training

Once you are in the docker container, you should already be inside a conda
environment, you can run nvidia-smi to confirm your compute resource and CUDA
version.

To start benchmarking:
```bash
cd /workspace/bert/
```

Edit the configuration in the scripts i.e. whether to profile, floating point
representation, number of GPUs, bach size, etc

for pre-training
```bash
vim scripts/run_pretraining.sh
```
for fine-tuning
```bash
vim scripts/run_squad.sh
```

then do
```bash
bash scripts/run_pretraining.sh 
```
or
```bash
bash scripts/run_squad.sh
```
to start the benchmark


### Measuring Throughput


### Profiling




# ImageClassification-ResNet-TensorFlow
## Description
This is a benchmarking suite for Resnet model under PyTorch.
The benchmarking source code is adopted from : [github link](https://github.com/tensorflow/models/tree/master/official/vision/image_classification)


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
>>Successfully tagged tbd-resnet-tensorflow:latest

### Running the Container
Next, start a docker container by running
```sh
bash ./start_container.sh <dataset dir>
```
This will prompt a shell in the container environment. You will need to provide a dataset directory to ./run_container.sh with the format specified in section [Preparing the dataset](###preparing-the-dataset).

Make sure you are inside of a the docker container by observing the command prompt:
>>(root) user@xxxxxxxxxxx:

### Preparing the Dataset
To use this benchmark, first download and decompress the [ImageNet LSVRC 2012 dataset](http://image-net.org/challenges/LSVRC/2012/). Note that you need to sign up for an account at image-net.org to download the dataset.
After decompressing, you should have training, bounding box annotations, and validation directories which look like this:
``` bash
$ DATA_DIR=.... # path to the ImageNet directory with train/val datasets
$ ls $DATA_DIR
$DATA_DIR/train:
n01440764
n01443537
n01484850
n01491361
...

$DATA_DIR/val:
n01440764
n01443537
n01484850
n01491361
...

$DATA_DIR/Annotation:
n01440764
n01443537
n01484850
n01491361
...
```

Note that you may need to rename the validation images to subfolders labeled as above, which can be done using the
following script under the validation directory:
https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

The training script requires TFRecord dataset instead of the raw image files. To generate the TFRecord files, run:

```bash
bash /app/source/dataset_util/convert_imagenet.sh /app/data
```
After processing, a directory containing the TFRecord files will be created under /app/data/TFRecords with the following format:
```bash
train-00000-of-01024
train-00001-of-01024
train-00002-of-01024
train-00003-of-01024
train-00004-of-01024
train-00005-of-01024
train-00006-of-01024
train-00007-of-01024
train-00008-of-01024
train-00009-of-01024
train-00010-of-01024
train-00011-of-01024
......
```
The size of the TfRecord files are around 144G

## Steps to Train and Profile
Note that in this benchmark we have chosen to use the default hyperparameters of the model, with a batch size of 32, using the number of GPUs available on the machine

### Running End to End Training

To perform end to end training, run the following command for training with default settings

```bash
CUDA_VISIBLE_DEVICES=<list of GPU device IDs> bash /app/source/scripts/run_train.sh  <arguments>
```

Where <list of GPU device IDs> is a comma separated list of GPUs to use for the training, and  <arguments to /app/source/main.py> are the arguments to the training script, if you want to change the choice of the default hyper-parameter values. The current default setting uses a batch size of 384, which is ideal for 4 GPUs each with 11GB of device memory, you should change the batch size according to your specific hardware setup.

To get more information on the various settings, run
```bash
bash /app/source/scripts/run_train.sh  --help
```
for the complete list of available values.

The training process takes 7 days on a single NVIDIA 2080Ti GPU, you will be prompted the training progress every 100 iterations by default.

### Measuring Throughput
To measure the single device raw training throughput of the benchmark for a list of batch sizes, run the following command.
```bash
CUDA_VISIBLE_DEVICES=<GPU device IDs>  bash /app/source/scripts/measure_throughput.sh  [stop_iteration]
```

Where [stop_iteration] is the iteration to stop recording the throughput data, default 1000. Note the program exists after stop_iteration.
You should see the same training progress as in the previous step, except it will run much shorter (several minutes). The the througput measurements are stored under "./throughput_data" 

### Profiling
There are two profiling modes (profilers) for this section, nvprof and ncu (nvidia Nsight System)

#### nvprof
nvprof is used to measure the compute utilization during the benchmark. To do so, the profiler generates the execution timeline of the benchmark, and users can use nvvp to extract the desired information.
For instructions on how to derive compute utilization from nvvp timelines, please refer to thie [link](https://github.com/UofT-EcoSystem/DNN-Training-Suite#visual-profiler)
To profile with nvprof, run:
```bash
CUDA_VISIBLE_DEVICES=<GPU device IDs>  bash /app/source/scripts/measure_compute_util.sh nvprof [start_iteration] [profile_iteration]
```
Where start_iteration (for example, 1000) is the iteration to start recording profiling information and profile_iteration (for example 200) is how many iterations to record.
The output of these the profiling steps are the NVPP timelines, which records kernel launch and memory transfer information during training. All NVVP files are stored under "profiled_data_nvprof", each experiment should output a file size of several hundred MBs.

#### ncu
ncu is used to measure the various core utilization during the benchmark. Currently, we measure 4 different types of cores, FMA, FP16, TensorCores, and SFU. These values are collected by the profiler through repeated executions of the CUDA kernels.
This script generates a csv file for each batch size, which records the different hardware performance metrics per each kernel invocation. These files are stored under ./profiled_data_ncu, and are several MBs large. The final result is post-processed by a script at /app/source/coreutil.py, which aggregates all collected performance counter information and reports the average core utilization values throughout the profiled period.
 
ncu is very slow, so please set [start_iteration] and [profile_iteration] to very small vales, between 1 and 10 (It is safe to do so since kernels are repeated for precise measurement).

To profile with ncu, first copy the file from [nsight log parser](/Core-Utilization-Analyzer), then run:
```bash
CUDA_VISIBLE_DEVICES=<list of GPU device IDs>  bash /app/source/scripts/measure_compute_util.sh ncu [start_iteration]  [profile_iteration]
# after the command is finished, each batch size choice generates a corresponding csv file, which can be parsed as
python coreutil.py profiled_data_ncu/batch_[X].csv
```


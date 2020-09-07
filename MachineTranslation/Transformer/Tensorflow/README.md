## Description
This benchmarks Transformer under TensorFlow. The benchmark source is adopted from [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)

The benchmark is performed using the following software environments CUDA version: 10.1 CUDNN version: 7.6 Ubuntu version: 18.04.

## 1 Setup
### 1.1 Getting Docker
This benchmark requires Docker. To do so, follow the steps [here](https://docs.docker.com/engine/install/).

### 1.2 Running the container
First, build the container with 
```
build_image.sh
```
This only has to be done once.

Then, start the container by with
```
run_container.sh
```
A shell inside the container will be spawned at the location `/scripts`. This directory contains the necessary scripts to perform training or benchmarking.

### 1.3 Obtaining dataset
From inside the container, run
```
bash download_iwslt15.sh /dataset
```
to download and extract the dataset. After this, the dataset folder should be structured as follows:
```
.
|-- train.en
|-- train.vi
|-- tst2012.en
|-- tst2012.vi
|-- tst2013.en
|-- tst2013.vi
|-- vocab.en
`-- vocab.vi
```
The dataset is around 32M. This process should take less than a minute for most cases.

## 2 Training
Inside the scripts folder, run
```
bash train.sh
```
Below is a training curve obtained from our run.
![](figures/train_en_vi.png)

## 3 Benchmarking
### 3.1 GPU Compute Utilization
Inside the scripts folder, run
```
bash profile.sh nvvp
```
to profile GPU compute utilization.

The results will be stored in `/scratch/measurements`. The filenames will follow the pattern of `profile_<timestamp>_bs<batch_size>.nvvp`. These results can be viewed with the Nvidia Visual Profiler. 

### 3.2 GPU Core Utilization
Inside the scripts folder, run
```
bash profile.sh nsight
```
to profile GPU core utilization.

The results will be stored in `/scratch/measurements`. The filenames will follow the pattern of `nsight_bs<batch_size>.csv`. You can analyze these results with the core utilization analyzer.


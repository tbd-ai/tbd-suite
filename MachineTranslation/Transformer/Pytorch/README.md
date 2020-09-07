## Description
This benchmarks Transformer under PyTorch. The benchmark source is adopted from [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

The benchmark is performed using the following software environments CUDA version: 10.1 CUDNN version: 7.6 Ubuntu version: 18.04.

## 1 Setup
### 1.1 Getting Docker
This benchmark requires Docker. To do so, follow the steps [here](https://docs.docker.com/engine/install/).

### 1.2 Running the container
First build the container with 
```
build_image.sh
```
This only has to be done once.

Then, start the container by with
```
run_container.sh
```
A shell inside the container will be spawned at the location `/scripts`. This directory contains the necessary scripts to perform training or benchmarking.

## 2 Dataset
Download and extract the dataset linked here: [https://s3.amazonaws.com/opennmt-trainingdata/wmt\_ende\_sp.tar.gz](https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp.tar.gz).

## 3 Training
Inside the scripts folder, run either:
```
bash opennmt-train.sh
```
to train the English-German model.

## 4 Benchmarking
### 4.1 GPU Compute Utilization
Inside the scripts folder, run
```
bash profile.sh nvprof
```
to profile GPU compute utilization.

NVProf will output its profiling results in `scratch/measurements`. You will find `nvvp` files with sizes larger than 100MB.

To parse and compute the results, use the NVidia Visual Profiler.


### 4.2 GPU Core Utilization
Inside the scripts folder, run
```
bash profile.sh nsight
```
to profile GPU core utilization.
The benchmarking results will be located inside `scratch/measurements` folder. The log files will be a few MBs in size.

To parse the results, use the [nsight log parser](https://github.com/UofT-EcoSystem/DNN-Training-Suite/tree/ybgao-temp/Core-Utilization-Analyzer).


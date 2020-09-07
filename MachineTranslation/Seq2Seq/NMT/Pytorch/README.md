## Description
This benchmarks GNMT under PyTorch. The benchmark source is adopted from [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT).

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
From inside the container, run
```
bash wmt16_en_de.sh
```
to download, extract, and process the dataset.

The dataset is around 1GB combined. Download is relatively slow due to the location of the server. Processing takes quite a while (~1hr). 

## 3 Training
Inside the scripts folder, run either:
```
bash train_ende.sh
```
to train the English-German or English-Vietnamese model respectively.

Below is a training curve obtained from our run (English-German). A total of 5 epochs were run.
![](figures/train_loss_en_de.png)

## 4 Benchmarking
Before beginning benchmarking, first create the folder in which benchmarking results are stored:
```
mkdir /scratch/measurements
```

### 4.1 GPU Compute Utilization
Inside the scripts folder, run
```
bash profile_ende.sh nvprof
```
to profile GPU compute utilization.

NVProf will output its profiling results in `scratch/measurements`. You will find `nvvp` files with sizes larger than 100MB.

To parse and compute the results, use the NVidia Visual Profiler.


### 4.2 GPU Core Utilization
Inside the scripts folder, run
```
bash profile_ende.sh nsight
```
to profile GPU core utilization.
The benchmarking results will be located inside `scratch/measurements` folder. The log files will be a few MBs in size.

To parse the results, use the [nsight log parser](/Core-Utilization-Analyzer).

### 4.3 Memory Utilization
Inside the scripts folder, run
```
mkdir /scratch/measurements/memory_csv_data
bash profile_ende.sh memory
```
to profile memory usage.

Memory utilization CSV will be written to `/scratch/measurements/memory_csv_data`.

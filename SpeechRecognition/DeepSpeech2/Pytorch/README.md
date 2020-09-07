## Description
This benchmark is for DeepSpeech2 under PyTorch. The benchmark source is adopted from [SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch).

The benchmark is performed using the following software environments CUDA version: 10.1 CUDNN version: 7.6 Ubuntu version: 18.04.

## 1 Setup
### 1.1 Getting Docker
This benchmark requires Docker. To do so, follow the steps [here](https://docs.docker.com/engine/install/).

### 1.2 Running the container
We first build the container with 
```
bash build_image.sh
```
This only has to be done once.

Then, start the container by with
```
bash run_container.sh
```
A shell inside the container will be spawned at the location `/scripts`. This directory contains the necessary scripts to perform training or benchmarking.

After starting the container, we install additional dependencies by running:
```
bash setup.sh
```

## 2 Dataset
Currently supports AN4, TEDLIUM, Voxforge, Common Voice and LibriSpeech. Scripts will setup the dataset and create manifest files used in data-loading. 
The scripts can be found in the scripts/data_downloader folder. Many of the scripts allow you to download the raw datasets separately if you choose so.

Our benchmark use the [LibriSpeech ASR corpus](http://www.openslr.org/12).The whole dataset is around 60G.

To download the LibriSpeech dataset, run
```
python librispeech.py
```

## 3 Training
To train the model, run
```
bash run_train.sh
```

## 4 Benchmarking
### 4.1 GPU Compute Utilization
Change the start/end iterations to 100 and 200 in `/source/deepspeech.pytorch/train_profiling.py`.
```
#<EcoSys> Add profiler start and end point
if i == 100:
   print('---------CUDA profile start---------')
   cuda.profile_start()
if i == 200:
   print('---------CUDA profile stop---------')
   cuda.profile_stop()    
#</EcoSys>
...
# <EcoSys> Break the training loop 
   if i == 200:
   sys.exit(0)
# <EcoSys>
```

Inside the scripts folder, run
```
bash run_profile.sh nvprof
```
to profile GPU compute utilization.

NVProf will output its profiling results in `scratch/measurements`.
The filenames will follow the pattern of `profile_<timestamp>_bs<batch_size>.nvvp`.
`nvvp` files with sizes from 100MB to 300MB.

To parse and compute the results, use the NVidia Visual Profiler.

### 4.2 GPU Core Utilization
Change the start/end iterations to 1 and 2 in `/source/deepspeech.pytorch/train_profiling.py`.
```
#<EcoSys> Add profiler start and end point
if i == 1:
   print('---------CUDA profile start---------')
   cuda.profile_start()
if i == 2:
   print('---------CUDA profile stop---------')
   cuda.profile_stop()    
#</EcoSys>
...
# <EcoSys> Break the training loop 
   if i == 2:
   sys.exit(0)
# <EcoSys>

```

Inside the scripts folder, run
```
bash run_profile.sh nsight
```
to profile GPU core utilization.
The benchmarking results will be located inside `scratch/measurements` folder.
The filenames will follow the pattern of `nsight_bs<batch_size>.csv`. 
The log files will be a few MBs in size.

To parse the results, use the [nsight log parser](/Core-Utilization-Analyzer).


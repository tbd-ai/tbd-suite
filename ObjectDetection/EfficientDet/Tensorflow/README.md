## Description
This benchmark is for EfficientDet under TensorFlow. The benchmark source is adopted from [google/automl](https://github.com/google/automl).

The benchmark is performed using the following software environments CUDA version: 10.1 CUDNN version: 7.6 Ubuntu version: 18.04.

## 1 Setup
### 1.1 Getting Docker
This benchmark requires Docker. To do so, follow the steps [here](https://docs.docker.com/engine/install/).

### 1.2 Running the container
First, build the container with 
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
From inside the container, run:
```
bash download_dataset.sh
```
to download the COCO dataset and convert it into tfrecords. 

## 3 Training
To train the model with the COCO dataset, run
```
bash train.sh
```

Since training takes a long time, we have evaluated the model with the provided checkpoint file for EfficientDet-D0. The results are listed below.
```
DONE (t=9.01s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.515
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.353
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.125
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.388
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.287
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.439
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.685
```
These agree with the advertized mAP on the model documentation.

## 4 Benchmarking
### 4.1 GPU Compute Utilization
Change the start/end iterations to 100 and 200 in `/source/automl/efficientdet/testhook.py`.

Inside the scripts folder, run
```
bash profile.sh nvprof
```
to profile GPU compute utilization.

NVProf will output its profiling results in `scratch/measurements`. You will find nvvp files with sizes larger than 100MB.

To parse and compute the results, use the NVidia Visual Profiler.


### 4.2 GPU Core Utilization
Change the start/end iterations to 1 and 2 in `/source/automl/efficientdet/testhook.py`.

Inside the scripts folder, run
```
bash profile.sh nsight
```
to profile GPU core utilization.

The benchmarking results will be located inside `scratch/measurements` folder. The log files will be a few MBs in size.

To parse the results, use the [nsight log parser](https://github.com/UofT-EcoSystem/DNN-Training-Suite/tree/ybgao-temp/Core-Utilization-Analyzer).

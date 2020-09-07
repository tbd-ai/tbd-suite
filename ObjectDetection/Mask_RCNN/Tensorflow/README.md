## Description
This benchmark is for MaskRCNN under TensorFlow. The benchmark source is adopted from [matterport/Mask\_RCNN](https://github.com/matterport/Mask_RCNN).

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
Run `download_coco.sh` from the `/scripts` folder. This will download and extract the COCO 2014 dataset to `/dataset`.

The total file size is around 20GB. 

You can expect the dataset structure to be similar to the following:

```
/dataset
  |-- annotations
  |-- train2014
  `-- val2014
```

## 3 Training
We first download the pretrained weights. Download the file [mask\_rcnn\_coco.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) and place it in `/source/Mask_RCNN`.

Then, we begin training the model with the COCO dataset:
```
bash switch.sh noprofile
bash train.sh
```

Since full training takes a long time, we instead evaluated the provided pretrained weights. Results are as follows:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.556
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.187
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.434
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.557
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.432
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.220
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.630
Prediction time: 109.67965626716614. Average 0.21935931253433227/image
Total time:  121.26065182685852
```

## 4 Benchmarking
### 4.1 GPU Compute Utilization
Inside the scripts folder, run
```
bash switch.sh nvprof
bash profile.sh nvprof
```
to profile GPU compute utilization.

NVProf will output its profiling results in `scratch/measurements`. You will find `nvvp` files with sizes larger than 100MB.

To parse and compute the results, use the NVidia Visual Profiler.

### 4.2 GPU Core Utilization
Inside the scripts folder, run
```
bash switch.sh nsight
bash profile.sh nsight
```
to profile GPU core utilization.
The benchmarking results will be located inside `scratch/measurements` folder. The log files will be a few MBs in size.

To parse the results, use the [nsight log parser](https://github.com/UofT-EcoSystem/DNN-Training-Suite/tree/ybgao-temp/Core-Utilization-Analyzer).

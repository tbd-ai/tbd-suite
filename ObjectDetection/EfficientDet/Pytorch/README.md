## Description
This benchmark is for EfficientDet under PyTorch. The benchmark source is adopted from [toandaominh1997/EfficientDet.Pytorch](https://github.com/toandaominh1997/EfficientDet.Pytorch).

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
Run the following to download VOC2007/2012 and COCO2017 datasets:
```
cd /source/EfficientDet.Pytorch;
# VOC
bash datasets/scripts/VOC2007.sh /dataset
bash datasets/scripts/VOC2012.sh /dataset
# COCO
bash datasets/scripts/COCO2017.sh /dataset
```

The folder structure should be similar to the following:
```
/dataset
|-- VOCdevkit
|   `-- VOC2007
|       |-- Annotations
|       |-- ImageSets
|       |-- JPEGImages
|       |-- SegmentationClass
|       `-- SegmentationObject
|-- annotations
`-- images
    |-- train2017
    `-- val2017
```

## 3 Training
To train the model with the COCO dataset, run
```
bash train_coco.sh
```
Since training takes a long time, I have evaluated the model with the VOC dataset using the provided VOC checkpoint file. Results are below:
```
mAP:
aeroplane: 0.6364755512448823
bicycle: 0.6577606651268633
bird: 0.44417639675906384
boat: 0.4534962414204517
bottle: 0.363183760181518
bus: 0.6579679721111347
car: 0.7468409370940656
cat: 0.6289737550182497
chair: 0.32843483468773405
cow: 0.5106001943385914
diningtable: 0.5184752909836183
dog: 0.5715819118424543
horse: 0.7089457625581794
motorbike: 0.6779383840025175
person: 0.7114742692464322
pottedplant: 0.29440760055417375
sheep: 0.5312860490456545
sofa: 0.5492348121828178
train: 0.7091399305835422
tvmonitor: 0.6029065601834924
avg mAP: 0.5651650439582717
```


## 4 Benchmarking
### 4.1 GPU Compute Utilization
Inside the scripts folder, run
```
bash profile.sh nvvp
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

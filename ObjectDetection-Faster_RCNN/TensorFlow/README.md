# ObjectDetection-Faster_RCNN-TensorFlow

This folder contains the TensorFlow implementation of the [Faster RCNN](https://github.com/tensorflow/models/tree/master/research/object_detection) model.

## Install Prerequisites

The libraries required to run the benchmark can be installed via apt-get:

```shell
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
```

## Prepare Dataset

The original directory of MXNet image classification contains scripts for all VOC07, VOC12 and COCO datasets. In our benchmark, we use only the VOC07 dataset.

To use our benchmark, download and convert the dataset to TFRecord format by:

``` shell
cd dataset
bash download-voc.sh
```

Note that the script will download and preprocess the VOC12 dataset as well.

## Prepare Pretrained Model

Download the pretrained ResNet-101 model:

``` shell
cd pretrained
bash download_pretrained.sh
```

## Run Training

To run the end-to-end training of Faster RCNN model with ResNet-101 kernel on VOC07 dataset, first change all the `<YOUR_PATH_TO_THE_TENSORFLOW_FASTER_RCNN_ROOT>` in `source/object_detection/samples/configs/faster_rcnn_resnet101_voc07.config` to the real path on your machine, then simply start the training by:

``` shell
cd scripts
bash faster-rcnn-voc07.sh
```

# Faster R-CNN

This package contains the MXNet implementation for [Faster R-CNN](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) model. The original implementation can be found [here](https://github.com/apache/incubator-mxnet/tree/v0.11.0/example/rcnn).

We use the end-to-end approach for training Faster R-CNN. Currently we support only ResNet-101 as the kernel model.

## Prepare Dataset

The original directory of MXNet image classification contains scripts for all VOC07, VOC12 and COCO datasets. In our benchmark, we use the VOC07 dataset.

To use our benchmark, download and convert the dataset to TFRecord format by:

``` shell
cd dataset
bash download-voc.sh
```

The script will download the VOC12 dataset as well.

## Prepare Pretrained Model

Download the pretrained ResNet-101 model by:

```shell
cd pretrained
bash download-pretrained-model.sh
```

## Run Training

Before running the training script, create a "model" directory in the rood directory of the benchmark:

```shell
mkdir model
```

To execute the end-to-end training of Faster R-CNN, first build the source:

```shell
cd source
make
```

Then start the training by running:

```shell
cd scripts
bash faster-rcnn-voc07.sh
```

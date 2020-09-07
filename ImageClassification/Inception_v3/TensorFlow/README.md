# ImageClassification-Inception-TensorFlow

This folder contains the TensorFlow implementation of the [inception-v3](https://github.com/tensorflow/models/tree/master/research/slim) model.

## Prepare Dataset

The original directory of TensorFlow image classification contains scripts for all Flowers, MNIST, cifar10 and imagenet datasets. In our benchmark, we use the imagenet1K dataset.

To use our benchmark, first prepare the dataset according to the following steps:

1, Download and decompress the [imagenet1K 2012 dataset](http://image-net.org/challenges/LSVRC/2012/). Note that you need to sign up for an account at image-net.org to download the dataset.
Rename and move your deceompressed directories so that the raw data folder looks like this:

``` shell
ImageNet1K/
	ImageNet_train/
		n01440764/
		n01443537/
		n01484850/
		...
	ImageNet_val/
		ILSVRC2012_val_00000001.JPEG
		ILSVRC2012_val_00000002.JPEG
		ILSVRC2012_val_00000003.JPEG
		...
```

2, Convert the raw data into TFRecord format:
``` shell
bash dataset/convert_imagenet.sh <path/to/your/ImageNet1K/folder>
```
This will generate a `TFRecords/` folder under `ImageNet1K/`, and you should see 1024 `train-xxxxx-of-yyyyy` files and 128 `validation-xxxxx-of-yyyyy` files inside. The fully generated files take about 142GB disk space.

## Run Training

We choose the following hyper-parameters for the inception-v3 models:

Learning rate: 0.1 for 30 epochs, 0.01 for 30 epochs, 0.001 for 40 epochs\
momentum: 0.9\
weight decay: 0.0001\
optimizer: sgd

We leave other hyper parameters as default.

Change the `DATASET_DIR` variable in `scripts/inception-imagenet.sh` (it should be the path to your `TFRecords/` folder), then start the training by:
``` shell
cd scripts
bash inception-imagenet.sh
```
To train the network with a different mini-batch size, change the `--batch_size` option in `scripts/inception-imagenet.sh`.

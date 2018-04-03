# ImageClassification-ResNet-MXNet

This folder contains the MXNet implementation of the [ResNet](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification) model.

## Prepare Dataset

The original directory of MXNet image classification contains scripts for all MNIST, cifar10 and imagenet datasets. In our benchmark, we use only the imagenet1K dataset.

To use our benchmark, first prepare the dataset according to the following steps:

1, Download and decompress the [imagenet1K 2012 dataset](http://image-net.org/challenges/LSVRC/2012/). Note that you need to sign up for an account at image-net.org to download the dataset. We use the training dataset, in another word, `ImageNet_train`. 
After decompressing, your dataset directory should look like this:

``` shell
$ DATA_DIR=.... # path to the ImageNet_train directory
$ ls $DATA_DIR
n01440764/
n01443537/
n01484850/
...
```

2, Generate the data format of RecordIO:

``` shell
python dataset/im2rec.py --list True --recursive True imagenet1k $DATA_DIR
python dataset/im2rec.py --resize 240 --quality 95 --num-thread 16 imagenet1k $DATA_DIR
```

You should have a file named `imagenet1k_train.rec` generated, the size of this file is about 46G.

## Run Training

We choose the following hyper-parameters for the ResNet-50 models:

Learning rate: 0.1 for 30 epochs, 0.01 for 30 epochs, 0.001 for 40 epochs\
momentum: 0.9\
weight decay: 0.0001\
optimizer: sgd

We leave other hyper parameters as default.

Change the `--data-train` option in `scripts/resnet-imagenet.sh`, then start the training by:
``` shell
cd scripts
bash resnet-imagenet.sh
```
Change the `--batch-size` option in `scripts/resnet-imagenet.sh` if you want to use a different mini-batch size.

To train a ResNet with more layers, add the `--num-layers` option. Here is an example of `scripts/resnet-imagenet.sh` to train ResNet-152:
``` shell
python ../source/train_imagenet.py --gpus 0 --batch-size 16 --image-shape 3,224,224 --num-epochs 100 --network resnet --num-layers 152 --data-train ... # the path to your generated .rec file
```


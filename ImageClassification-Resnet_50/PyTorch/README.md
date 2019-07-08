# ImageClassification-ResNet-PyTorch

This folder contains the PyTorch implementation of the [ResNet](https://github.com/pytorch/examples/tree/master/imagenet) model.

## Steps to Configure Machine

Install Python 3.6, [PyTorch v0.4.0](https://github.com/pytorch/pytorch/tree/v0.4.0), [Cuda 9.2](https://developer.nvidia.com/cuda-downloads) and [Cudnn 7.1](https://developer.nvidia.com/cudnn)

Ensure that nvprof is in your path, and that you are using nvprof from Cuda 9.2
```bash
$ which nvprof
/usr/local/cuda-9.2/bin/nvprof
```

Install other python packages

```bash
pip install -r source/requirements.gpu-cu92.txt
```

## Steps to Prepare Dataset

To use this benchmark, first download and decompress the [ImageNet LSVRC 2012 dataset](http://image-net.org/challenges/LSVRC/2012/). Note that you need to sign up for an account at image-net.org to download the dataset. 
After decompressing, you should have training and validation directories which look like this:

``` bash
$ DATA_DIR=.... # path to the ImageNet directory with train/val datasets
$ ls $DATA_DIR
$DATA_DIR/train:
n01440764
n01443537
n01484850
n01491361
...

$DATA_DIR/val:
n01440764
n01443537
n01484850
n01491361
...
```
Note that you may need to rename the validation images to subfolders labeled as above, which can be done using the 
following script:

https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Steps to Train and Profile

Run the `scripts/benchmark-resnet50.sh` script to either train or profile the model. Note that when using the `--profile` or 
`--profile-fp32` modes, the model will not train to completion; instead, it will exit upon 100 iterations of profiling.

```bash
# Run options:
bash scripts/benchmark-resnet50.sh <path_to_imagenet_dataset> # Train
bash scripts/benchmark-resnet50.sh <path_to_imagenet_dataset> --profile # Profile for compute utilization
bash scripts/benchmark-resnet50.sh <path_to_imagenet_dataset> --profile-fp32 # Profile for fp32 utilization
```

Note that in this benchmark we have chosen to use the default hyperparameters of the model, with a batch size of 32.


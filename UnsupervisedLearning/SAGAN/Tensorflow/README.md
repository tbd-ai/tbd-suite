# Tensorflow - Self-Attention Generative Adversarial Networks

## Introduction

This directory contains the [Self-Attention Generative Adversarial Networks](https://github.com/brain-research/self-attention-gan) benchmark. 

This benchmark is 
based on the Self-Attention GAN model with the ImageNet LSVRC 2012 dataset. 
## Dependencies

Install Python 3.6, Cuda 10.0, Cudnn 7 and Tensorflow 1.13.2

Ensure that nvprof is in your path, and you are using nvprof from Cuda 10.0
```
which nvprof
```

From the root of the benchmark, execute the following commands to set up dependencies. 

```
pip install -r source/requirements.gpu-cu100.txt
```


## Downloading the Data
```
# change to dataset directory
cd dataset
```

Download the LSVRC 2012 dataset from http://image-net.org/challenges/LSVRC/2012/index.

Your directory structure should be like this

```
├── dateset
    ├── convert-imagenet-2012.sh
    ├── convert_imagenet_to_records.py
    ├── ILSVRC2012_img_train.tar
```

Execute the following commands to convert the ImageNet2012 data to .tfrecord (Converting process may take few hours).

```
bash convert-imagenet-2012.sh
```

Your directory structure should be like this after execution:

```
├── dateset
    ├── convert-imagenet-2012.sh
    ├── convert_imagenet_to_records.py
    ├── data 
        ├── n01440764
            ├── n01440764_18.JPEG
            ├── n01440764_36.JPEG
            ├── n01440764_37.JPEG
            ├── ...
        ├── n01443537
        ├── n01443537
        ├── ......
```

## Change the tensorflow file to profile SAGAN

Replace the training file of GAN in tensorflow by training.py in this repo. Typically, one could find the training.py file of tensorflow in: 

```
/usr/local/anaconda3/envs/python36/lib/python3.6/site-packages/tensorflow/contrib/training/python/training/training.py
```

Add ecosys code (between 'EcoSys' comments) to training.py file at your local. Code can be found at train() function in source/training.py (this repo)

## Running the Benchmark

From the root of the benckmark, execute the following commands to run the benchmark. (Full training process may take more than a week to reach the reported fid score on single 1080ti GPU)

```
cd scripts
CUDA_VISIBLE_DEVICES=0,1,2,3 bash benchmark-imagenet-2012.sh
```

You may need to adjust batch size to run on machines with different GPU memory.

## Reference

```
@article{Han18,
  author    = {Han Zhang and
               Ian J. Goodfellow and
               Dimitris N. Metaxas and
               Augustus Odena},
  title     = {Self-Attention Generative Adversarial Networks},
  year      = {2018},
  journal = {arXiv:1805.08318},
}
```

## Changes

* Added profiler start and end point.

* Changed some python2 code to python3.

* Changed pathing to organize output files better.


# Tensorflow - Improved Training of Wasserstein GANs

## Introduction

This directory contains the [_Improved Training of Wasserstein GANs_](https://github.com/igul222/improved_wgan_training) 
benchmark. The repository contains multiple experiments, but this benchmark is 
based on the ResNet model with the [imagenet-small](http://image-net.org/small/download.php) dataset. 

(Python 2.7 only)

## Downloading the Data

From the root of the benchmark, execute the following commands to download the data. 

```
cd dataset
bash download-imagenet-small.sh
```

## Running the Benchmark

From the root of the benckmark, execute the following commands to run the benchmark. 

```
cd script
bash benchmark-imagenet-small.sh
```

* You may need to adjust batch size to run on machines with less GPU memory.

## Reference

```
@article{DBLP:journals/corr/GulrajaniAADC17,
  author    = {Ishaan Gulrajani and
               Faruk Ahmed and
               Mart{\'{\i}}n Arjovsky and
               Vincent Dumoulin and
               Aaron C. Courville},
  title     = {Improved Training of Wasserstein GANs},
  journal   = {CoRR},
  volume    = {abs/1704.00028},
  year      = {2017},
  url       = {http://arxiv.org/abs/1704.00028},
  archivePrefix = {arXiv},
  eprint    = {1704.00028},
  timestamp = {Wed, 07 Jun 2017 14:42:35 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/GulrajaniAADC17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Changes

* Added profiler start and end point.

* Added wall-clock time to output.

* Changed pathing to organize output files better.


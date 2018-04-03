# Sockeye

## Introduction

This folder contains the [_Sockeye_](https://github.com/awslabs/sockeye) benchmark.
It is developed by the [_Amazon Web Services_](https://aws.amazon.com/).
The local copy is based on version [_1.5_](https://github.com/awslabs/sockeye/tree/1.5).
_Sockeye_ is the benchmark on machine translation that supports three different models

* RNN-based Sequence-to-Sequence

* Transformer

* Convolutional Sequence-to-Sequence

Here, we chose the hyperparameters specifically to run the first model. 
The dataset _IWSLT15_ was taken from the [_Tensorflow NMT_](https://github.com/tensorflow/nmt/tree/tf-1.2) repository.
Due to the fact that _Sockeye_ cannot process the raw dataset directly, 
we slightly modified the original dataset so that empty lines have been removed.

## Reference

```
@article{Sockeye:17,
   author = {Hieber, Felix and Domhan, Tobias and Denkowski, Michael
           and Vilar, David and Sokolov, Artem, and Clifton, Ann and Post, Matt},
    title = "{Sockeye: A Toolkit for Neural Machine Translation}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1712.05690},
 primaryClass = "cs.CL",
 keywords = {Computer Science - Computation and Language,
             Computer Science - Learning,
             Statistics - Machine Learning},
     year = 2017,
    month = dec,
      url = {https://arxiv.org/abs/1712.05690}
}
```

## Sockeye 1.5 - Changelog

* [Changed the beam size from 5 to 10.](./source/sockeye-1.5/sockeye/constants.py#L133)

* [Changed the measurement frequency from 50 to 100 updates.](./source/sockeye-1.5/sockeye/constants.py#L119)

* [Rewrote how vocabularies are built from vocabulary files.](./source/sockeye-1.5/sockeye/train.py#L56)

* [Added the profiler start and end point.](./source/sockeye-1.5/sockeye/training.py#L301)

* [Changed the bucketing policy to be consistent with the NMT model.](./source/sockeye-1.5/sockeye/data_io.py#L67)

* [Ignore the line that has only one ".".](./source/sockeye-1.5/sockeye/data_io.py#L245)

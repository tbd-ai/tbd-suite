# A3C - Tensorflow Model

## Introduction

This folder contains the Tensorflow version of the A3C Benchmark.
It is part of the [_Denny Britz's repository_](https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient/a3c).
The local copy is based on TensorFlow version [_tf-1.3_](https://github.com/tensorflow/tensorflow/tree/r1.3).

(Python 3 only)

## Running

```
bash ./scripts/run.py
```

The visualization of the training progress is monitored with Tensorboard:

```
tensorboard --logdir=/tmp/a3c
```

## A3C Changelog

* [Added logs for training informations.](./source/train.py#L26)
* [Added the GPU device instead of the CPU.](./source/train.py#L88)
* [Added the tensorflow config session options.](./source/train.py#L133)

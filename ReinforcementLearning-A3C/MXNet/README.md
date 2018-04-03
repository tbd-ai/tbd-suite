# A3C - MXNet Model

## Introduction

This folder contains the MXNet version of the A3C Benchmark.
This implementation is part of the [_Apache MXNet (incubating) for Deep Learning_](https://github.com/apache/incubator-mxnet). It is a generalization of the original algorithm since it uses batch_size threads for each worker instead of the original 1 thread.
The local copy is based on the  [_original version_](https://github.com/apache/incubator-mxnet/tree/master/example/reinforcement-learning/a3c).

## A3C Changelog

* [Added the numba module call to enable the profiling preference.](./source/a3c.py#L31)
* [Added the profiling start and end preference.](./source/a3c.py#L62)
* [Added the profiling check during the training.](./source/a3c.py#L160)

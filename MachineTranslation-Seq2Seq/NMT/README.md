# NMT

## Introduction

This folder contains the [_NMT_](https://github.com/tensorflow/nmt) benchmark.
It is developed by the [_Google Tensorflow_](https://github.com/tensorflow).
The local copy is based on version [_tf-1.2_](https://github.com/tensorflow/nmt/tree/tf-1.2).
_NMT_ is the benchmark on machine translation that uses Recurrent Neural Networks
and has been claimed to achieve comparable training results to that of [_GNMT_](https://arxiv.org/abs/1609.08144).
The dataset _IWSLT15_ was taken from the original _NMT_ repository.

## NMT-tf-1.2 Changelog

* [Added the measurements on total number of samples.](./source/nmt-tf-1.2/nmt/train.py#L337)

* [Added the profiler start and end point.](./source/nmt-tf-1.2/nmt/train.py#L359)

* [Increase the total number of samples by batch size.](./source/nmt-tf-1.2/nmt/train.py#L401)

* [Added samples per second to the log file.](./source/nmt-tf-1.2/nmt/train.py#L413)


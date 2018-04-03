# DeepSpeech 2

## Introduction

This folder contains the [_DeepSpeech2_](https://arxiv.org/abs/1512.02595) benchmark.
It is developed by [_Baidu_](http://www.baidu.com/).
The local copy is based on example code in MXNet repo. [_example_](https://github.com/apache/incubator-mxnet/tree/master/example/speech_recognition).
_DeepSpeech2_ is the benchmark on Speech Recognition

## Prerequisites
* Install sox for converting *.flac files to *.wav
```bash
  sudo apt-get install sox
```
* For setting up DS2 on MXNet, Warp-CTC need to be installed first. Please follow the steps given [here](https://github.com/UofT-EcoSystem/DNN-Training-Suite/tree/master/SpeechRecognition-DeepSpeech2/MXNet/source#install-baidu-warpctc).

## Reference

```
@article{DBLP:journals/corr/AmodeiABCCCCCCD15,
  author    = {Dario Amodei and
               Rishita Anubhai and
               Eric Battenberg and
               Carl Case and
               Jared Casper and
               Bryan Catanzaro and
               Jingdong Chen and
               Mike Chrzanowski and
               Adam Coates and
               Greg Diamos and
               Erich Elsen and
               Jesse Engel and
               Linxi Fan and
               Christopher Fougner and
               Tony Han and
               Awni Y. Hannun and
               Billy Jun and
               Patrick LeGresley and
               Libby Lin and
               Sharan Narang and
               Andrew Y. Ng and
               Sherjil Ozair and
               Ryan Prenger and
               Jonathan Raiman and
               Sanjeev Satheesh and
               David Seetapun and
               Shubho Sengupta and
               Yi Wang and
               Zhiqian Wang and
               Chong Wang and
               Bo Xiao and
               Dani Yogatama and
               Jun Zhan and
               Zhenyao Zhu},
  title     = {Deep Speech 2: End-to-End Speech Recognition in English and Mandarin},
  journal   = {CoRR},
  volume    = {abs/1512.02595},
  year      = {2015},
  url       = {http://arxiv.org/abs/1512.02595},
  archivePrefix = {arXiv},
  eprint    = {1512.02595},
  timestamp = {Wed, 07 Jun 2017 14:42:02 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/AmodeiABCCCCCCD15},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## DeepSpeech 2 - Changelog

* [Change from tensorboard to tensorboardX.](./source/train.py#L26)

* [Importing helper functions for CUDA profiling.](./source/train.py#L32)

* [Add profiler start and end point.](./source/train.py#L150)


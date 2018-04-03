# Tensor2Tensor - Transformer

## Introduction

This directory contains the [_Tensor2tensor_](https://github.com/tensorflow/tensor2tensor) 
transformer benchmark. Tensor2tensor is a project that includes a variety of datasets and tasks.
In this local copy, we focus on machine translation using the transformer model on the WMT 2014
English to German corpus.

## Reference

```
@article{DBLP:journals/corr/VaswaniSPUJGKP17,
  author    = {Ashish Vaswani and
               Noam Shazeer and
               Niki Parmar and
               Jakob Uszkoreit and
               Llion Jones and
               Aidan N. Gomez and
               Lukasz Kaiser and
               Illia Polosukhin},
  title     = {Attention Is All You Need},
  journal   = {CoRR},
  volume    = {abs/1706.03762},
  year      = {2017},
  url       = {http://arxiv.org/abs/1706.03762},
  archivePrefix = {arXiv},
  eprint    = {1706.03762},
  timestamp = {Mon, 03 Jul 2017 13:29:02 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/VaswaniSPUJGKP17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Notes

* When running any script, do so from the directory containing the script.
* Computing the actual bleu score requires mosesdecoder. You can download this from github. Place this in your home directory as ~/mosesdecoder/
  * Since the time that this benchmark was made, the tensor2tensor library was changed to make accurate bleu score computation much simpler.

## Instructions For Computing Accurate BLEU Scores

1. In the scripts directory...
2. ./average-checkpoints.sh
3. ./decode-transformer.sh
4. ./calculate-bleu-from-decodes.sh
5. BLEU score is the first number shown on the output line.

## Changes

* Changed the main/executable python files so that they include tensor2tensor in their path. This was needed because tensor2tensor is originally intended to be installed.
* [Added utils/profile_monitor.py to training.](./source/tensor2tensor/tensor2tensor/utils/profile_monitor.py)
* Added scripts to simplify the process of computing the BLEU score.


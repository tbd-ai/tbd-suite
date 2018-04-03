# MachineTranslation-Transformer

## Introduction

This folder contains machine translation benchmarks that use the transformer model.

The transformer model is a recent machine translation model that achieved state of the art results, despite not using any recurrence. Instead, it just uses attention layers. Since attention layers are primarily composed of matrix multiplication and other relatively GPU-efficient operations compared to RNN operations, the transformer model achieves superior GPU utilization and occupancy.

## References

The transformer model was introduced in this paper:

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

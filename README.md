# DNN-Training-Suite

## Introduction

This repository contains the benchmark suite for DNN training.

The benchmarks was categorized according to what their application and model (e.g. MachineTranslation-Seq2Seq, 
MachineTranslation-Transformer, ...). 
Each category contains a `README.md` that points to related resources (e.g. papers, tutorials, ...).

Each benchmark contains the following:

* `source`: Source files of the DNN benchmarks, with the suffix indicating its version (e.g. git branch or tag name).
Each source folder includes `requirements.gpu-cu80.txt` that contains all the packages needed on a `CUDA 8.0` machine.
Please note that benchmark source files might be changed in order for them to run properly or make fair comparisons. All the changes will be highlighted using the following format:

```python
# <EcoSys> A brief description of what the changes are.
Changes go here ...
# </EcoSys>
```

* `dataset`: Scripts that download the dataset, and follows the naming convention `download-dataset_name.sh`.

* `scripts`: Scripts that run the benchmark on certain dataset, and follows the naming convention `benchmark-dataset.sh`.
If no arguments are given, the script will help you run the benchmark. 
If command argument `--profile` is given, the script will do profiling on CUDA kernels.
If command argument `--profile-fp32` is given, the script will do profiling on utilization on floating point units.
(Please note that you will need to input those `.nvvp` files to NVidia Visual Profiler to see the profiling results.)

* `README.md`: An introduction on what the benchmark is, 
and contains changelog that records changes that were made to the benchmark. 
If the benchmark was published as a paper, then proper bibtex will also be included.

## Instructions

* Setup Virtual Environment: To avoid conflicts between different benchmarks, 
we strongly recommend that you setup a Python virtual environment for each benchmark separately. 
Please note that many benchmarks have a restriction on the Python version used.

```
virtualenv --system-site-packages -p python2 <virtual-env-name> # python 2.x
virtualenv --system-site-packages -p python3 <virtual-env-name> # python 3.x

source <virtual-env-name>/bin/activate # Activate the virtual environment.
```

* Install the benchmark prerequisite: 

```
cd <benchmark-name>/source/; pip install -r requirements.gpu-cu80.txt
```

* Download the benchmark dataset:

```
cd <benchmark-name>/dataset/
chmod 700 ./download-<dataset_name>.sh
./download-<dataset_name>.sh
```

* Use the script files to run or profile the benchmark:

```
cd <benchmark-name>/scripts/
chmod 700 ./<benchmark_name>-<dataset-name>.sh
./<benchmark_name>-<dataset_name>.sh
```

## Known Issues

* The hyperparameters have been tuned specifically to fit into the GPU that has `8 GB` memory. If you receive an error message that informs you of Out-Of-Memory error, please carefully check `nvidia-smi` and make sure that no one else is using the machine.

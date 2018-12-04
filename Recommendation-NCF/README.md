# 1. Problem
This task benchmarks recommendation with implicit feedback on the [MovieLens 20 Million (ml-20m) dataset](https://grouplens.org/datasets/movielens/20m/) with a [Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569) model.
The model trains on binary information about whether or not a user interacted with a specific item.

# 2. Directions
### Steps to configure machine


Install Python 3.6, [PyTorch v0.4.0](https://github.com/pytorch/pytorch/tree/v0.4.0), [Cuda 9.2](https://developer.nvidia.com/cuda-downloads) and [Cudnn 7.1](https://developer.nvidia.com/cudnn)

Ensure that nvprof is in your path, and that you are using nvprof from Cuda 9.2
```bash
which nvprof
```


Install `unzip` and `curl`

```bash
sudo apt-get install unzip curl
```

Install other python packages

```bash
cd PyTorch
pip install -r source/requirements.gpu-cu92.txt
```

### Steps to download and verify data

You can download and verify the dataset by running the `download_dataset.sh` and `verify_dataset.sh` scripts in the parent directory:

```bash
# Creates ml-20.zip
source dataset/download_dataset.sh
# Confirms the MD5 checksum of ml-20.zip
source dataset/verify_dataset.sh
```

### Steps to run and profile


Run the `benchmark-ncf.sh` script to either train or profile the model. Note that when using the `--profile` or 
`--profile-fp32` modes, the model will not train to completion; instead, it will exit upon 100 iterations of profiling.

```bash
# Run options:
source scripts/benchmark-ncf.sh # Train
source scripts/benchmark-ncf.sh --profile # Profile for compute utilization
source scripts/benchmark-ncf.sh --profile-fp32 # Profile for fp32 utilization
```
# 3. Pretrained Model

A pretrained model (trained using the default settings available in this source code) can be found on [our group's Google Drive server](https://drive.google.com/open?id=1IaTqcCAWYhFsbvL4bnlJ2oEdGjP9OV7M).
To use this model, the [PyTorch documentation](https://pytorch.org/docs/master/notes/serialization.html) recommends 
using the following API calls:
```bash
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
```

# 4. Dataset/Environment
### Publication/Attribution
Harper, F. M. & Konstan, J. A. (2015), 'The MovieLens Datasets: History and Context', ACM Trans. Interact. Intell. Syst. 5(4), 19:1--19:19.

### Data preprocessing

1. Unzip
2. Remove users with less than 20 reviews
3. Create training and test data separation described below

### Training and test data separation
Positive training examples are all but the last item each user rated.
Negative training examples are randomly selected from the unrated items for each user.

The last item each user rated is used as a positive example in the test set.
A fixed set of 999 unrated items are also selected to calculate hit rate at 10 for predicting the test item.

### Training data order
Data is traversed randomly with 4 negative examples selected on average for every positive example.

# 5. Model
### Publication/Attribution
Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569). In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

The author's original code is available at [hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering).

### MLPerf repository
The MLPerf repository used as a basis for this model is available at [mlperf/reference/tree/master/recommendation/pytorch](https://github.com/mlperf/reference/tree/master/recommendation/pytorch)
# 6. Quality
### Quality metric
Hit rate at 10 (HR@10) with 999 negative items.

### Quality target
HR@10: 0.635

### Evaluation frequency
After every epoch through the training data.

### Evaluation thoroughness

Every users last item rated, i.e. all held out positive examples.

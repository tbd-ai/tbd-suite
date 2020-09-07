# ImageClassification-ResNet-CNTK

This folder contains the CNTK python implementation of the ResNet model. The original official repo is [here](https://github.com/Microsoft/CNTK/tree/release/2.0/Examples/Image/Classification/ResNet).

## Prepare Dataset

The original directory of CNTK ResNet contains code for cifar10 and imagenet datasets. In our benchmark, we use the imagenet1K dataset.

To use our benchmark, first prepare the dataset according to the following steps:

1, Download and decompress the [imagenet1K 2012 dataset](http://image-net.org/challenges/LSVRC/2012/). Note that you need to sign up for an account at image-net.org to download the dataset.
Rename and move your deceompressed directories so that the raw data folder looks like this:

``` shell
ImageNet1K/
	ImageNet_train/
		n01440764/
		n01443537/
		n01484850/
		...
	ImageNet_val/
		ILSVRC2012_val_00000001.JPEG
		ILSVRC2012_val_00000002.JPEG
		ILSVRC2012_val_00000003.JPEG
		...
```

2, Copy `dataset/Read_labelclsloc.py` & `dataset/create_train_map.py` to `ImageNet1K/ImageNet_train/`, and `dataset/ILSVRC2012_validation_ground_truth.txt` & `dataset/create_val_map.py` to `ImageNet1K/ImageNet_val/`

3, Generate train_map.txt and val_map.txt by:
```shell
DATA_DIR=<ABS_PATH_TO_ImageNet1K/ImageNet_train/>
cd $DATA_DIR
python3.5 create_train_map.py $DATA_DIR
cd ../ImageNet_val
python create_val_map.py
```

## Get CNTK Executable
Currently we provide CNTK BrainScript supported by CNTK-2.0. We will update to latest version of CNTK in future release.

To train the model with CNTK BrainScript, obtain the CNTK executable by following:\
1, Install open MPI by `sudo apt-get install libopenmpi-dev`;\
2, Download and decompress CNTK-2.0 GPU version for Linux [here](https://cntk.ai/dllg-2.0.html). To use other version, check [here](https://github.com/Microsoft/CNTK/releases) (Please download the release, not the source code);\
3, Add dependency libs to the environment by:
```shell
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:<PATH/TO/cntk/cntk/dependencies/lib>
```
4, You should find the cntk executable in `<PATH/TO/cntk/cntk/bin/cntk>`.

## Run Training

To start the training of ResNet-50 with mini-batch size x with CNTK BrainScript, you need to:\
1, Change the `DataDir` variable in `source/BrainScript/ResNet50_ImageNet1K.cntk` to `<ABS_PATH_TO_ImageNet1K/ImageNet_train/>`;\
2, Modify the `minibatchSize` variable to x;\
3, start the training by:
```shell
<PATH/TO/cntk/cntk/bin/cntk> configFile=ResNet50_ImageNet1K.cntk
```
To train ResNet-152, change the `DataDir` & `minibatchSize` variables accordingly, then start the training by:
```shell
<PATH/TO/cntk/cntk/bin/cntk> configFile=ResNet152_ImageNet1K.cntk
```

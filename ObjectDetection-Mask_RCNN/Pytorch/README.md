Object detection and segmentation. Metrics are mask and box mAP.

# Prerequisite
* Make sure pytorch installed properly (at least pytorch 0.4.1, which contains caffe2)
* Install dtrx to unzip .zip and .tgz file
```bash
  sudo apt-get install dtrx
```
* Install nvidia-docker2 to run the model (if version 1 not installed before): https://github.com/NVIDIA/nvidia-docker
```bash
  sudo apt-get install -y nvidia-docker2
  sudo pkill -SIGHUP dockerd
  # optional, test if nvidia-docker2 installed successfully
  docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```
# Prepare Dataset
This model uses coco2014 dataset to train and validate.
```bash
  $ cd dataset
  $ bash download-coco.sh
```

# Run Training
Build the nvidia-docker2 container.
```bash
  $ cd ../source
  $ sudo docker build -t detectron .
```

Run the docker container and mount the data appropriately

```bash
  $ sudo nvidia-docker run -v `realpath ../dataset/coco/`:/packages/detectron/lib/datasets/data/coco -v `realpath ../nvprof_data/`:/packages/detectron/nvprof_data -v `realpath ../scripts/`:/packages/detectron/scripts -it detectron /bin/bash
```

Start training or profiling
```bash
  $ bash scripts/benchmark-mask-rcnn-coco2014.sh # Train
  $ bash scripts/benchmark-mask-rcnn-coco2014.sh --profile # Profile for compute utilization
  $ bash scripts/benchmark-mask-rcnn-coco2014.sh --profile-fp32 # Profile for fp32 utilization
```

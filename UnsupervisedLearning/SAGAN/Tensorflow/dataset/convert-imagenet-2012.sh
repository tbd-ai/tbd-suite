# Unzip tar files
mkdir data
tar -xvf ILSVRC2012_img_train.tar -C data
rm ILSVRC2012_img_train.tar

# Build the TFRecords version of the ImageNet data.
BUILD_SCRIPT="./convert_imagenet_to_records.py"

sudo python3 "${BUILD_SCRIPT}"

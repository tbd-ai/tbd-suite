# download dataset
wget https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
# download weights
wget https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl


# verify dataset
if md5sum -c hashes.md5
then
  echo "PASSED"
else
  echo "DOWNLOAD FAILED"
  exit
fi


# suppose you want to save those files at ./coco
mkdir ./coco
mv coco_annotations_minival.tgz ./coco
mv train2014.zip ./coco
mv val2014.zip ./coco
mv annotations_trainval2014.zip ./coco

mv R-50.pkl ./coco

cd ./coco

# extract dataset
dtrx --one=here coco_annotations_minival.tgz
dtrx --one=here annotations_trainval2014.zip
mv annotations.1/* annotations/

dtrx train2014.zip
mv train2014/ coco_train2014/
dtrx val2014.zip
mv val2014/ coco_val2014/

rm -rf *.zip
rm -rf *.tgz


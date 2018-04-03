declare -a arr=("VOCtrainval_06-Nov-2007.tar")
for i in "${arr[@]}"
do
    if ! [ -e $i ]
    then
        echo $i "not found, downloading"
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/$i
    fi
done

# the result is VOCdevkit/VOC2012
voc2012="VOCtrainval_11-May-2012.tar"
if ! [ -e $voc2012 ]
then
    echo $voc2012 "not found, downloading"
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/$voc2012
fi

tar -xvf VOCtrainval_06-Nov-2007.tar

export PYTHONPATH=../source/
python create_pascal_tf_record.py \
    --label_map_path=pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2007 --set=train \
    --output_path=pascal_train_2007.record
python create_pascal_tf_record.py \
    --label_map_path=pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2007 --set=val \
    --output_path=pascal_val_2007.record

tar -xvf VOCtrainval_11-May-2012.tar
python create_pascal_tf_record.py \
    --label_map_path=pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=train \
    --output_path=pascal_train_2012.record
python create_pascal_tf_record.py \
    --label_map_path=pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=val \
    --output_path=pascal_val_2012.record


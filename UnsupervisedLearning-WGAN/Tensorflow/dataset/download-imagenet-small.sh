wget http://image-net.org/small/train_64x64.tar -P ../imnet_data/
wget http://image-net.org/small/valid_64x64.tar -P ../imnet_data/
tar -xvf ../imnet_data/train_64x64.tar -C ../imnet_data/
tar -xvf ../imnet_data/valid_64x64.tar -C ../imnet_data/
rm ../imnet_data/train_64x64.tar
rm ../imnet_data/valid_64x64.tar

curl https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip --output model.zip
unzip model.zip -d ../dataset/model 
cd ../dataset/model 
cd `ls | head`
mv ./* ../
cd ../../../scripts 
rm model.zip
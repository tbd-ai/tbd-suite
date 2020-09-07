apt install -y libgl1;

# install cocoapi
pip3 install pycocotools;

# install maskrcnn dependencies
cd /source/Mask_RCNN;
pip3 install -r requirements.txt;
python3 setup.py install;




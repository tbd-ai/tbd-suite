cd /source/Mask_RCNN/mrcnn;

if [ "$1" == "nvprof" ]; then
	rm model.py;
	ln -s model.py.profile_nvprof model.py;
elif [ "$1" == "nsight" ]; then
	rm model.py;
	ln -s model.py.profile_nsight model.py;
elif [ "$1" == "noprofile" ]; then
	rm model.py;
	ln -s model.py.noprofile model.py;
fi

# uninstall previous version
pip3 uninstall -y mrcnn;
rm -rf /source/Mask_RCNN/build;

# build and install new version
cd /source/Mask_RCNN/;
python3 setup.py install;

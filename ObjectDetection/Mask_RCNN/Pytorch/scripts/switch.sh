cd /source/detectron2_repo/detectron2/engine;

if [ "$1" == "nvprof" ]; then
	rm train_loop.py;
	ln -s train_loop.py.profile_nvprof train_loop.py;
elif [ "$1" == "nsight" ]; then
	rm train_loop.py;
	ln -s train_loop.py.profile_nsight train_loop.py;
elif [ "$1" == "memory" ]; then
	rm train_loop.py;
	ln -s train_loop.py.profile_memory train_loop.py;
elif [ "$1" == "noprofile" ]; then
	rm train_loop.py;
	ln -s train_loop.py.noprofile train_loop.py;
fi

# uninstall previous version
pip3 uninstall -y detectron2;
rm -r /source/detectron2_repo/build;

# build and install new version
cd /source/detectron2_repo/;
python3 setup.py install;

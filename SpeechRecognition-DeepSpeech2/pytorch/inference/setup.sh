yes 'y' | sudo apt-get remove unscd; yes 'y' | sudo apt-get install python-pip; pip install sox wget; yes 'y' | sudo apt-get install sox libsox-fmt-mp3

cd ../dataset
if [ -d "LibriSpeech_dataset" ]
then
	echo "\n\nLibrispeech folder found, skipping download.\n\n"
	sleep 2
else
	echo "\n\nDownloading clean_dev, (est. 25 min, space req 15G)...\n\n"
	sleep 2
	sh download_dataset.sh clean_dev
fi
cd ../inference	

if [ "${1}" = "cuda" ]
then
	VARIANT="cuda_"
	yes 'y' | sudo add-apt-repository ppa:graphics-drivers/ppa
	yes 'y' | sudo apt-get update
	yes 'y' | sudo apt-get install nvidia
	yes 'y' | sudo apt-get install cuda-drivers
	yes 'y' | sudo apt-get install htop
else
	VARIANT=""
fi

cd ../docker
yes 'y' | sh install_${VARIANT}docker.sh

GROUP="docker"
sudo usermod -a -G $GROUP $USER
newgrp $GROUP << END						# Need to run docker related items as a user in this group!
echo "\n\nBuilding Docker Image (up to 8min)\n\n"
sleep 2
yes 'y' | sh build_${VARIANT}docker.sh
END

cd ../inference
echo "Ready to run:\n\tnewgrp ${GROUP}\n\tsh ../docker/run_${VARIANT}dev.sh"


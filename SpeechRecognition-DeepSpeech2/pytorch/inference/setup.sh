yes 'y' | sudo apt-get remove unscd; yes 'y' | sudo apt-get install python-pip; pip install sox wget; yes 'y' | sudo apt-get install sox libsox-fmt-mp3
	
cd ../dataset
#sh download_dataset.sh clean_dev
cd ../inference

echo "what"
if [ "${1}" = "cuda" ]
then
	VARIANT="cuda_"
else
	VARIANT=""
fi

cd ../docker
yes 'y' | sh install_${VARIANT}docker.sh
sudo usermod -a -G docker $USER

yes 'y' | sh build_${VARIANT}docker.sh

cd ../inference
echo "Ready to run: sh ../docker/run_${VARIANT}dev.sh"


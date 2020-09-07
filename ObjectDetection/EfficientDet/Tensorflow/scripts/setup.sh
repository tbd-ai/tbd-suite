
# update packages
apt update;
apt install -y python3 python3-pip;
pip3 install --upgrade pip;

SOURCES_ROOT=/source
DATASET_DIR="/dataset"

# install python requirements
cd ${SOURCES_ROOT}/automl/efficientdet; \
	pip3 install --no-cache-dir -r requirements.txt

# install cocoapi
cd ${SOURCES_ROOT}/cocoapi/PythonAPI; \
	python3 setup.py install


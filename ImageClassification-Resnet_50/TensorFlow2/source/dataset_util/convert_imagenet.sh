# Create the output and temporary directories.
DATA_DIR="${1%/}"
SCRATCH_DIR="${DATA_DIR}"

# Download the ImageNet data.
LABELS_FILE="./imagenet_lsvrc_2015_synsets.txt"

# Note the locations of the train and validation data.
TRAIN_DIRECTORY="${SCRATCH_DIR}/train/"
VALIDATION_DIRECTORY="${SCRATCH_DIR}/val/"


BOUNDING_BOX_DIR="${SCRATCH_DIR}/Annotation/"
BOUNDING_BOX_FILE="./imagenet_2012_bounding_boxes.csv"

# Build the TFRecords version of the ImageNet data.
BUILD_SCRIPT="./build_imagenet_data.py"
OUTPUT_DIRECTORY="${DATA_DIR}/TFRecords"
IMAGENET_METADATA_FILE="./imagenet_metadata.txt"

mkdir -p "${OUTPUT_DIRECTORY}"

python "${BUILD_SCRIPT}" \
  --train_directory="${TRAIN_DIRECTORY}" \
  --validation_directory="${VALIDATION_DIRECTORY}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --imagenet_metadata_file="${IMAGENET_METADATA_FILE}" \
  --labels_file="${LABELS_FILE}" \
  --bounding_box_file="${BOUNDING_BOX_FILE}"

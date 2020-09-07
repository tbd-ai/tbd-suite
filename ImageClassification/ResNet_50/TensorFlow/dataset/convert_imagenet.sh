# Create the output and temporary directories.
DATA_DIR="${1%/}"
SCRATCH_DIR="${DATA_DIR}/ImageNet"

# Download the ImageNet data.
LABELS_FILE="./dataset/imagenet_lsvrc_2015_synsets.txt"

# Note the locations of the train and validation data.
TRAIN_DIRECTORY="${SCRATCH_DIR}_train/"
VALIDATION_DIRECTORY="${SCRATCH_DIR}_val/"

# Preprocess the validation data by moving the images into the appropriate
# sub-directory based on the label (synset) of the image.
echo "Organizing the validation data into sub-directories."
PREPROCESS_VAL_SCRIPT="./dataset/preprocess_imagenet_validation_data.py"
VAL_LABELS_FILE="./dataset/imagenet_2012_validation_synset_labels.txt"

"${PREPROCESS_VAL_SCRIPT}" "${VALIDATION_DIRECTORY}" "${VAL_LABELS_FILE}"

# Convert the XML files for bounding box annotations into a single CSV.
echo "Extracting bounding box information from XML."
BOUNDING_BOX_SCRIPT="./dataset/process_bounding_boxes.py"
BOUNDING_BOX_FILE="./dataset/imagenet_2012_bounding_boxes.csv"
BOUNDING_BOX_DIR="${SCRATCH_DIR}_bounding_boxes/"

"${BOUNDING_BOX_SCRIPT}" "${BOUNDING_BOX_DIR}" "${LABELS_FILE}" \
 | sort >"${BOUNDING_BOX_FILE}"
echo "Finished downloading and preprocessing the ImageNet data."

# Build the TFRecords version of the ImageNet data.
BUILD_SCRIPT="./dataset/build_imagenet_data.py"
OUTPUT_DIRECTORY="${DATA_DIR}/TFRecords"
IMAGENET_METADATA_FILE="./dataset/imagenet_metadata.txt"

mkdir -p "${OUTPUT_DIRECTORY}"

sudo python3 "${BUILD_SCRIPT}" \
  --train_directory="${TRAIN_DIRECTORY}" \
  --validation_directory="${VALIDATION_DIRECTORY}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --imagenet_metadata_file="${IMAGENET_METADATA_FILE}" \
  --labels_file="${LABELS_FILE}" \
  --bounding_box_file="${BOUNDING_BOX_FILE}"

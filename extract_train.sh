#!/bin/bash
mkdir -p /tmp/ILSVRC2012_train_extracted

# Initialize a label mapping file
label_mapping_file="/tmp/label_mapping.txt"
echo "Generating label mapping..."
> "$label_mapping_file"  # Empty the file if it exists

# Extract the main tar file
tar -xf /scratch/shareddata/dldata/imagenet/ILSVRC2012_img_train.tar -C /tmp/ILSVRC2012_train_extracted

# Initialize label counter
label_id=0

# Process each .tar file individually
for tar_file in /tmp/ILSVRC2012_train_extracted/*.tar; do
    foldername=$(basename "$tar_file" .tar)

    # Write the mapping for the current class
    echo "$foldername $label_id" >> "$label_mapping_file"
    label_id=$((label_id + 1))

    # Create folder for extracted contents
    mkdir -p "/tmp/ILSVRC2012_train_extracted/$foldername"
    tar -xf "$tar_file" -C "/tmp/ILSVRC2012_train_extracted/$foldername"
    rm "$tar_file"
done

echo "Label mapping generated at $label_mapping_file"

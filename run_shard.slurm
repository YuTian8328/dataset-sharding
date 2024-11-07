#!/bin/bash
#SBATCH --job-name=shard_imagenet           # Job name
#SBATCH --output=shard_imagenet_%j.log       # Output log file (%j will be replaced by the job ID)
#SBATCH --error=shard_imagenet_%j.err        # Error log file
#SBATCH --ntasks=1                           # Number of tasks (set to 1 for a single script)
#SBATCH --cpus-per-task=8                    # Number of CPU cores per task 
#SBATCH --mem=32G                            # Total memory 
#SBATCH --time=06:00:00                      # Time limit 
#SBATCH --partition=gpu-h100-80g              #Partition 

# Load required module
module load scicomp-python-env 

./extract_train.sh

mkdir /tmp/sharded

# Define input, output directories, and the number of images per shard
INPUT_DIR="/tmp/ILSVRC2012_train_extracted"
OUTPUT_DIR="/tmp/sharded"
NUM_IMAGES_PER_SHARD=60000

# Run the sharding script
python -u shard_imagenet.py --input_dir "$INPUT_DIR" \
                         --output_dir "$OUTPUT_DIR" \
                       --num_images_per_shard "$NUM_IMAGES_PER_SHARD" \
                       --label_mapping_file "/tmp/label_mapping.txt"

# Copy back to your own folder
cp /tmp/sharded/* /scratch/rse/build-shards/sharded
#!/bin/bash
#SBATCH --job-name=imagenet-training         # Job name
#SBATCH --output=imagenet_training_%j.out     # Standard output and error log
#SBATCH --error=imagenet_training_%j.err
#SBATCH --ntasks=1                            # Number of tasks (1 task for single-node)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task (adjust as needed)
#SBATCH --gpus=1                              # Number of GPUs per task (adjust as needed)
# SBATCH --time=02:00:00                       # Time limit hrs:min:sec, if gpu-debug, don't set this
#SBATCH --partition=gpu-debug                   # Specify the GPU partition (adjust based on your system)

# Load necessary module
module load scicomp-python-env


# Run the training script
python -u train_with_sharded.py --shard_path "/scratch/rse/build-shards/sharded/imagenet-{0000..0021}.tar" --batch_size 64 --num_workers 8 --learning_rate 0.0005 --epochs 10

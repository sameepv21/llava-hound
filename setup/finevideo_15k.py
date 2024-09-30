import os
import random
import shutil

source_dir = "/scratch/svani/data/finevideo/data"
destination_dir = "/scratch/svani/data/finevideo/finevideo_15k"
os.makedirs(destination_dir, exist_ok=True)

# Get list of all parquet files in the source directory
parquet_files = [f for f in os.listdir(source_dir) if f.endswith('.parquet')]

# Randomly sample 468 parquet files
sampled_files = random.sample(parquet_files, 468)

# Copy the sampled files to the destination directory
for file in sampled_files:
    shutil.move(os.path.join(source_dir, file), os.path.join(destination_dir, file))
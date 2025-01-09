#!/bin/bash

# Define dataset_str and model_str arrays
datasets=("bitcoinotc" "reddit-title" "email-eu")
models=("evolvegcn" "gcnrngru" "roland")

# Base directory for logs
base_dir="logs/xai_results"

# Create the directory structure
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    # Create the directory for each combination of dataset and model
    dir_path="$base_dir/$dataset/$model"
    mkdir -p "$dir_path"
    echo "Created directory: $dir_path"
  done
done

# Notify completion
echo "All directories have been created successfully."
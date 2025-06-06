#!/bin/bash
DIR=$1
echo $DIR
for data in mnist fmnist kmnist emnist cifar10; do
  echo "Dataset:" $data
  uv run analysis_model_evaluation.py --dataset $data --dropout -o $DIR
done

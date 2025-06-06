#!/bin/bash
DIR=$1
echo $DIR
for data in fmnist mnist kmnist emnist cifar10; do
  echo "Dataset:" $data
  uv run analysis_model_evaluation.py --dataset $data -o $DIR
done

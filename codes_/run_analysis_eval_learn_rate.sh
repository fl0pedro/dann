#!/bin/bash
DIR=$1
echo $DIR
for lr in 0.01 0.0001; do
  echo "Dataset:" $data
  uv run analysis_model_evaluation.py --learning-rate $lr -o $DIR
done

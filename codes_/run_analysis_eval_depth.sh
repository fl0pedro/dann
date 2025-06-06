#!/bin/bash
DIR=$1
echo $DIR
for nl in 2 3; do
  echo "Number of layers:" $nl
  uv run analysis_model_evaluation.py --num-layers nl -o $DIR
done

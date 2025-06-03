#!/bin/bash
DIRPARAMS=$1
echo $DIRPARAMS
seq_flag=0 # sequential flag
noise_flag=0 # noise flag
drop_flag=0 # dropout flag
data=fmnist # dataset
lr=0.001 # learning rate
es_flag=0 # early stop flag
for nl in 2 3; do
  echo "Number of layers:" $nl
  uv run analysis_model_evaluation.py $DIRPARAMS $data $seq_flag $noise_flag $drop_flag $nl $lr $es_flag
done

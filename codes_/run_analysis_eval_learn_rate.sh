#!/bin/bash
DIRPARAMS=$1
echo $DIRPARAMS
seq_flag=0 # sequential flag
noise_flag=0 # noise flag
drop_flag=0 # droput flag
nl=1 # number of layers (2 hidden layers)
data=fmnist # dataset
es_flag=0 # early stop flag
for lr in 0.01 0.0001; do
  echo "Dataset:" $data
  python analysis_model_evaluation.py $DIRPARAMS $data $seq_flag $noise_flag $drop_flag $nl $lr $es_flag
done

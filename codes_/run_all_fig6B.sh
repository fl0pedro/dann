#!/bin/bash
DIRPARAMS=$1
GPU_ID=0 # the GPU to be used
seq_flag=0 # sequnetial learning flag (no-sequential)
estop_flag=0 # early stopping flag
data=fmnist # dataset to train on
nl=1 # number of dendro-somatic layers (2 hidden layers)
nsyn=16 # number of synapses
drop_flag=0 # dropout flag for vanilla model
rate=0 # rate of dropout
lr=0.001 # adam learning rate
for sigma in 0.25 0.5 0.75 1.0; do
  for t in $(seq 1 5); do
    for m in 10 11; do #$(seq 0 11); do
      for d in 1 2 4 8 16 32 64; do
        for s in 32 64 128 256 512; do
          echo "Trial:" $t "Model:" $m "dends:" $d "soma:" $s
          python main.py $GPU_ID $seq_flag $estop_flag $t $m $sigma $data $d $s $nl $nsyn $drop_flag $rate $lr $DIRPARAMS
        done
      done
    done
  done
done

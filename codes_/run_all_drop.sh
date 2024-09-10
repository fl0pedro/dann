#!/bin/bash
DIRPARAMS=$1
echo $DIRPARAMS
GPU_ID=0 # the GPU to be used
seq_flag=0 # sequnetial learning flag (no-sequential)
estop_flag=0 # early stopping flag
m=3 # set the model to vanilla_ann
sigma=0.0 # noise
nl=1 # number of dendrosomatic layers (2 hidden layers)
nsyn=16 # number of synapses
drop_flag=1 # dropout flag for vanilla model
lr=0.001 # adam learning rate
for data in mnist kmnist emnist cifar10; do
  for rate in 0.2 0.5 0.8; do
    for t in $(seq 1 5); do
      for d in 1 2 4 8 16 32 64; do
        for s in 32 64 128 256 512; do
          echo "Trial:" $t "Model:" $m "dends:" $d "soma:" $s
          python main.py $GPU_ID $seq_flag $estop_flag $t $m $sigma $data $d $s $nl $nsyn $drop_flag $rate $lr $DIRPARAMS
        done
      done
    done
  done
done

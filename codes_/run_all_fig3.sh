#!/bin/bash
DIRPARAMS=$1
echo $DIRPARAMS
# if the fig2 script has run, no need for fmnist afain. If not, add it the for loop for data
for data in mnist fmnist kmnist emnist cifar10; do
  for t in $(seq 1 5); do
    for m in $(seq 0 11); do
      for d in 1 2 4 8 16 32 64; do
        for s in 32 64 128 256 512; do
          echo "Trial:" $t "Model:" $m "dends:" $d "soma:" $s
          uv run main.py --gpu --trial $t --model $m --dataset $data -d $d -s $s -o
        done
      done
    done
  done
done


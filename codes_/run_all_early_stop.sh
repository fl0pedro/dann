#!/bin/bash
DIRPARAMS=$1
echo $DIRPARAMS
# if the fig2 script has run, no need for fmnist afain. If not, add it the for loop for data
for d in 1 2 4 8 16 32 64; do
  for s in 32 64 128 256 512; do
    for data in fmnist cifar10; do #mnist fmnist kmnist emnist cifar10; do
      for t in $(seq 1 5); do
        for m in $(seq 0 11); do
          echo "Trial:" $t "Model:" $m "dends:" $d "soma:" $s
          uv run main.py --gpu --trial $t --model $m --dataset $data --early-stop -d $d -s $s
        done
      done
    done
  done
done


#!/bin/bash
DIRPARAMS=$1
echo $DIRPARAMS
for data in mnist kmnist emnist cifar10; do
  for rate in 0.2 0.5 0.8; do
    for t in $(seq 1 5); do
      for d in 1 2 4 8 16 32 64; do
        for s in 32 64 128 256 512; do
          echo "Trial:" $t "Model:" $m "dends:" $d "soma:" $s
          uv run main.py --gpu --trial $t --model 3 --dataset $data --drop-rate $rate -d $d -s $s -o
        done
      done
    done
  done
done

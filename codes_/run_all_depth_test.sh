#!/bin/bash
DIR=$1
echo $DIR
for t in $(seq 1 5); do
  for m in $(seq 0 11); do
    for d in 8 16 32 64; do
      for s in 256 512; do
        echo "Trial:" $t "Model:" $m "dends:" $d "soma:" $s
        uv run main.py --gpu --trial $t --model $m -d $d -s $s --num-layers 2 -o $DIR
      done
    done
  done
done

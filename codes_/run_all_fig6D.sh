#!/bin/bash
DIR=$1
echo $DIR
for t in $(seq 1 5); do
  for m in 10 11; do # $(seq 0 11); do
    for d in 1 2 4 8 16 32 64; do
      for s in 32 64 128 256 512; do
        echo "Trial:" $t "Model:" $m "dends:" $d "soma:" $s
          uv run main.py --gpu --trial $t --model $m --sequential -d $d -s $s -o $DIR
      done
    done
  done
done

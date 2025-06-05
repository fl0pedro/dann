#!/bin/bash
#SBATCH --output=slurm/%x_%j.out 
#SBATCH --error=slurm/%x_%j.err 
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00 
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL 
#SBATCH --mail-user=f.assmuth@fz-juelich.de
#SBATCH --job-name=dANN

MAX_JOBS=4
JOBS=0
for x in $(ls run*.sh); do 
  DIR=$(basename $x) 
  sh $x "all_out/$DIR" 1>slurm/$DIR.out 2>slurm/$DIR.err &
  ((JOBS++))
  if (( JOBS % MAX_JOBS == 0 )); then
    wait
  fi
done
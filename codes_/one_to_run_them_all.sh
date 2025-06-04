#!/bin/bash
#SBATCH --output=slurm/%x_%j.out 
#SBATCH --error=slurm/%x_%j.err 
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00 
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL 
#SBATCH --mail-user=f.assmuth@fz-juelich.de
#SBATCH --job-name=dANN

for x in $(ls run*.sh); do sh $x all_out; done

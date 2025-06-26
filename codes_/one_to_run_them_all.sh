#!/bin/bash
#SBATCH --output=slurm/%x_%j.out 
#SBATCH --error=slurm/%x_%j.err 
#SBATCH --time=48:00:00 
#SBATCH --partition=pgi15-cpu
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL 
#SBATCH --mail-user=f.assmuth@fz-juelich.de
#SBATCH --job-name=dANN

uv run one_to_run_them_all.py -o all_out_2 -w $(nproc) --backend tensorflow
#!/bin/bash
#SBATCH --output=slurm/%x_%j.out 
#SBATCH --error=slurm/%x_%j.err 
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00 
#SBATCH --nodelist=pgi15-cpu2
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL 
#SBATCH --mail-user=f.assmuth@fz-juelich.de
#SBATCH --job-name=dANN

uv sync
for x in $(ls run*.sh); do 
    base=$(basename "$x" .sh)

    echo "running $x..."
    sh $x all_out 1> "$base.out" 2> "$base.err" &
done

wait
echo "all done!"

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

MAX_JOBS=1000
TOTAL_JOBS=0
COMPLETED_JOBS=0
DIR=slurm

run_job() {
  local args="$1"
  local name="$2"
  ((TOTAL_JOBS++))
  (
    uv run main.py $args > "$DIR/${name}.out" 2> "$DIR/${name}.err"
    ((COMPLETED_JOBS++))
  ) &
}

wait_for_slot() {
  while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
    echo "Running: $(jobs -r | wc -l), Completed: $COMPLETED_JOBS/$TOTAL_JOBS"
    sleep 2
  done
}

for t in $(seq 1 5); do
  for m in $(seq 0 11); do
    for d in 8 16 32 64; do
      for s in 256 512; do
        name="nl2_t$t_m$m_d$d_s$s"
        args="--trial $t --model $m -d $d -s $s --num-layers 2 -o $DIR"
        wait_for_slot
        run_job "$args" "$name"
      done
    done
  done
done

for nl in 2 3; do
  for t in $(seq 1 5); do
    for m in $(seq 0 11); do
      for d in 8 16 32 64; do
        for s in 256 512; do
          name="nl$nl_t$t_m$m_d$d_s$s"
          args="--trial $t --model $m -d $d -s $s --num-layers $nl -o $DIR"
          wait_for_slot
          run_job "$args" "$name"
        done
      done
    done
  done
done

for data in mnist kmnist emnist cifar10; do
  for rate in 0.2 0.5 0.8; do
    for t in $(seq 1 5); do
      for d in 1 2 4 8 16 32 64; do
        for s in 32 64 128 256 512; do
          name="drop_t$t_d$d_s$s_data$data_r$rate"
          args="--trial $t --model 3 --dataset $data --drop-rate $rate -d $d -s $s -o $DIR"
          wait_for_slot
          run_job "$args" "$name"
        done
      done
    done
  done
done

for d in 1 2 4 8 16 32 64; do
  for s in 32 64 128 256 512; do
    for data in fmnist cifar10; do
      for t in $(seq 1 5); do
        for m in $(seq 0 11); do
          name="es_t$t_m$m_d$d_s$s_data$data"
          args="--trial $t --model $m --dataset $data --early-stop -d $d -s $s -o $DIR"
          wait_for_slot
          run_job "$args" "$name"
        done
      done
    done
  done
done

for t in $(seq 1 5); do
  for m in $(seq 0 11); do
    for d in 1 2 4 8 16 32 64; do
      for s in 32 64 128 256 512; do
        name="base_t$t_m$m_d$d_s$s"
        args="--trial $t --model $m -d $d -s $s -o $DIR"
        wait_for_slot
        run_job "$args" "$name"
      done
    done
  done
done

for data in mnist fmnist kmnist emnist cifar10; do
  for t in $(seq 1 5); do
    for m in $(seq 0 11); do
      for d in 1 2 4 8 16 32 64; do
        for s in 32 64 128 256 512; do
          name="all_t$t_m$m_d$d_s$s_data$data"
          args="--trial $t --model $m --dataset $data -d $d -s $s -o $DIR"
          wait_for_slot
          run_job "$args" "$name"
        done
      done
    done
  done
done

for sigma in 0.25 0.5 0.75 1.0; do
  for t in $(seq 1 5); do
    for m in 10 11; do
      for d in 1 2 4 8 16 32 64; do
        for s in 32 64 128 256 512; do
          name="sig$sigma_t$t_m$m_d$d_s$s"
          args="--trial $t --model $m --sigma $sigma -d $d -s $s -o $DIR"
          wait_for_slot
          run_job "$args" "$name"
        done
      done
    done
  done
done

for t in $(seq 1 5); do
  for m in 10 11; do
    for d in 1 2 4 8 16 32 64; do
      for s in 32 64 128 256 512; do
        name="seq_t$t_m$m_d$d_s$s"
        args="--trial $t --model $m --sequential -d $d -s $s -o $DIR"
        wait_for_slot
        run_job "$args" "$name"
      done
    done
  done
done

for lr in 0.01 0.0001; do
  for t in $(seq 1 5); do
    for m in $(seq 0 11); do
      for d in 1 2 4 8 16 32 64; do
        for s in 32 64 128 256 512; do
          name="lr$lr_t$t_m$m_d$d_s$s"
          args="--trial $t --model $m --dataset cifar10 --learning-rate $lr -d $d -s $s -o $DIR"
          wait_for_slot
          run_job "$args" "$name"
        done
      done
    done
  done
done

echo "Waiting for all $TOTAL_JOBS jobs to finish..."
wait
echo "All $COMPLETED_JOBS jobs completed!"
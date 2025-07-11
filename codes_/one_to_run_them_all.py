from itertools import product
from tqdm import tqdm
from random import shuffle
from concurrent.futures import ProcessPoolExecutor
import main as run_model
import analysis_model_evaluation as analyze
import argparse
import os

job_file = "job_args.txt"

def run_main_calls(output_dir, max_workers=4, backend="jax", gpu=False):
    jobs = []
    if os.path.exists('main_' + job_file):
        with open('main_' + job_file, 'r') as f:
            jobs.extend([l.split() for l in f.read().split('\n')])
    else:
        all_data = ["mnist", "fmnist", "kmnist", "emnist", "cifar10"]

        for nl in [2, 3]:
            for t, m, d, s in product(range(1, 6), range(12), [8, 16, 32, 64], [256, 512]):
                args = f"--trial {t} --model {m} -d {d} -s {s} --num-layers {nl}".split()
                jobs.append(args)

        rates = [0.2, 0.5, 0.8]
        for data, rate in product(all_data, rates):
            for t, d, s in product(range(1, 6), [1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512]):
                args = f"--trial {t} --model 3 --dataset {data} --drop-rate {rate} -d {d} -s {s}".split()
                jobs.append(args)

        for data in ["fmnist", "cifar10"]:
            for d, s, t, m in product([1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512], range(1, 6), range(12)):
                args = f"--trial {t} --model {m} --dataset {data} --early-stop -d {d} -s {s}".split()
                jobs.append(args)

        for t, m, d, s in product(range(1, 6), range(12), [1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512]):
            args = f"--trial {t} --model {m} -d {d} -s {s}".split()
            jobs.append(args)

        for data, t, m, d, s in product(all_data, range(1, 6), range(12), [1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512]):
            args = f"--trial {t} --model {m} --dataset {data} -d {d} -s {s}".split()
            jobs.append(args)

        for sigma in [0.25, 0.5, 0.75, 1.0]:
            for t, m, d, s in product(range(1, 6), [10, 11], [1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512]):
                args = f"--trial {t} --model {m} --sigma {sigma} -d {d} -s {s}".split()
                jobs.append(args)

        for t, m, d, s in product(range(1, 6), [10, 11], [1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512]):
            args = f"--trial {t} --model {m} --sequential -d {d} -s {s}".split()
            jobs.append(args)

        for lr in [0.01, 0.0001]:
            for t, m, d, s in product(range(1, 6), range(12), [1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512]):
                args = f"--trial {t} --model {m} --dataset cifar10 --learning-rate {lr} -d {d} -s {s}".split()
                jobs.append(args)
        
        with open('main_' + job_file, 'w') as f:
            f.write('\n'.join([' '.join(j) for j in jobs]))

    # Parallel execution
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        jobs = [j + f"-o {output_dir} --backend {backend} {'--gpu' if gpu else ''}".split() for j in jobs]
        list(tqdm(executor.map(run_model.main, shuffle(jobs)), total=len(jobs), desc="Running models"))
    
    os.remove(job_file)


def run_analysis_calls(output_dir, max_workers=4, backend="torch", gpu=False):
    jobs = []
    if os.path.exists('analysis_' + job_file):
        with open('analysis_' + job_file, 'r') as f:
            jobs.extend([l.split() for l in f.read().split('\n')])
    else:
        all_data = ["mnist", "fmnist", "kmnist", "emnist", "cifar10"]

        for l in [2, 3]:
            args = f"--num-layers {l}".split()
            jobs.append(args)

        for data in all_data:
            args = f"--dataset {data} --dropout".split()
            jobs.append(args)

        for lr in [0.01, 0.0001]:
            args = f"--learning-rate {lr}".split()
            jobs.append(args)

        for data in all_data:
            args = f"--dataset {data}".split()
            jobs.append(args)

        with open('analysis_' + job_file, 'w') as f:
            f.write('\n'.join([' '.join(j) for j in jobs]))
            
    # Parallel execution
    with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1) as executor:
        jobs = [j + f"-o {output_dir} --backend {backend} {'--gpu' if gpu else ''} -w {max_workers//len(jobs) or 1} -s".split() for j in jobs]
        shuffle(jobs)
        list(tqdm(executor.map(analyze.main, jobs), total=len(jobs), desc="Running analysis"))
    
    os.remove(job_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output")
    parser.add_argument("-w", "--workers", type=int, default=4)
    parser.add_argument("-b", "--backend", choices=["tensorflow", "torch", "jax"], default="jax")
    parser.add_argument("--gpu", action="store_const", const="1", default="")
    args = parser.parse_args()
    
    run_main_calls(args.output, args.workers, args.backend, args.gpu)
    run_analysis_calls(args.output, args.workers, args.backend, args.gpu)
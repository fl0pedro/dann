from itertools import product
from tqdm import tqdm
import main as run_model
import analysis_model_evaluation as analyze

def run_main_py_calls(output_dir):
    all_data = ["mnist", "fmnist", "kmnist", "emnist", "cifar10"]

    for nl in [2, 3]:
        loop = product(range(1, 6), range(12), [8, 16, 32, 64], [256, 512])
        for t, m, d, s in tqdm(loop, desc=f"num-layers={nl}", total=5 * 12 * 4 * 2):
            args = f"--trial {t} --model {m} -d {d} -s {s} --num-layers {nl} -o {output_dir}".split()
            run_model(args)

    rates = [0.2, 0.5, 0.8]
    for data, rate in tqdm(product(all_data, rates), desc="dropout datasets", total=len(datasets)*len(rates)):
        for t, d, s in product(range(1, 6), [1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512]):
            args = f"--trial {t} --model 3 --dataset {data} --drop-rate {rate} -d {d} -s {s} -o {output_dir}".split()
            run_model(args)

    for data in tqdm(["fmnist", "cifar10"], desc="early-stop datasets"):
        for d, s, t, m in product([1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512], range(1, 6), range(12)):
            args = f"--trial {t} --model {m} --dataset {data} --early-stop -d {d} -s {s} -o {output_dir}".split()
            run_model(args)

    loop = product(range(1, 6), range(12), [1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512])
    for t, m, d, s in tqdm(loop, desc="baseline grid", total=5 * 12 * 7 * 5):
        args = f"--trial {t} --model {m} -d {d} -s {s} -o {output_dir}".split()
        run_model(args)

    loop = product(all_data, range(1, 6), range(12), [1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512])
    for data, t, m, d, s in tqdm(loop, desc="all datasets", total=5 * 5 * 12 * 7 * 5):
        args = f"--trial {t} --model {m} --dataset {data} -d {d} -s {s} -o {output_dir}".split()
        run_model(args)

    for sigma in [0.25, 0.5, 0.75, 1.0]:
        loop = product(range(1, 6), [10, 11], [1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512])
        for t, m, d, s in tqdm(loop, desc=f"noise sigma={sigma}", total=5 * 2 * 7 * 5):
            args = f"--trial {t} --model {m} --sigma {sigma} -d {d} -s {s} -o {output_dir}".split()
            run_model(args)

    loop = product(range(1, 6), [10, 11], [1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512])
    for t, m, d, s in tqdm(loop, desc="sequential models", total=5 * 2 * 7 * 5):
        args = f"--trial {t} --model {m} --sequential -d {d} -s {s} -o {output_dir}".split()
        run_model(args)

    for lr in [0.01, 0.0001]:
        loop = product(range(1, 6), range(12), [1, 2, 4, 8, 16, 32, 64], [32, 64, 128, 256, 512])
        for t, m, d, s in tqdm(loop, desc=f"lr={lr}", total=5 * 12 * 7 * 5):
            args = f"--trial {t} --model {m} --dataset cifar10 --learning-rate {lr} -d {d} -s {s} -o {output_dir}".split()
            run_model(args)

def run_analysis_calls(output_dir):
    all_data = ["mnist", "fmnist", "kmnist", "emnist", "cifar10"]
    
    for nl in tqdm([2, 3], desc="num-layers"):
        args = f"--num-layers {l} -o {output_dir}"
        analyze(args)

    for data in tqdm(all_data, desc="dropout datasets"):
        args = f"--dataset {data} --dropout -o {output_dir}"
        analyze(args)

    for lr in tqdm([0.01, 0.0001], desc="learning rate"):
        args = f"--learning-rate {lr} -o {output_dir}"
        analyze(args)

    for data in tqdm(all_data, desc="full dataset eval"):
        args = f"--dataset {data} -o {output_dir}"
        analyze(args)

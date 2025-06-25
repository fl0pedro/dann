#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:00:15 2021.
@author: spiros and assmuth
"""
import os
import time
import pickle
import pathlib
import numpy as np
import pandas as pd
import argparse
from main import main as run_model
from itertools import product

from utils import num_trainable_params, get_power_of_10
from opt import get_model_idx

def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="dirname")
    parser.add_argument("--gpu", action="store_const", const="1", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--learning-rate", dest="lr", type=float, default=1e-3)
    parser.add_argument("--dataset", choices=["mnist", "fmnist", "kmnist", "emnist", "cifar10"], default="fmnist")
    parser.add_argument("--backend", choices=["tensorflow", "torch", "jax"], default="jax")
    parser.add_argument("-w", "--worker", type=int, default=4)
    return parser.parse_args(args)

def main(args: list[str] | None = None):
    args = parse_args(args)
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    t1 = time.time()

    if not args.dropout:
        models = [
            "dend_ann_random", "dend_ann_local_rfs", "dend_ann_global_rfs", "dend_ann_all_to_all",
            "vanilla_ann", "vanilla_ann_local_rfs", "vanilla_ann_global_rfs", "vanilla_ann_random",
            "sparse_ann", "sparse_ann_local_rfs", "sparse_ann_global_rfs", "sparse_ann_all_to_all",
        ]
    else:
        models = [
            "dend_ann_random", "dend_ann_local_rfs", "dend_ann_global_rfs",
            "vanilla_ann", "vanilla_ann_dropout_0.2", "vanilla_ann_dropout_0.5", "vanilla_ann_dropout_0.8"
        ]

    somata = [1<<i for i in range(5, 10)]
    dendrites = [1<<i for i in range(7)]
    sigmas = [0.0] if not args.noise else np.linspace(0,1,5).tolist()
    trials = range(1, 6)
    tag = "_sequential" if args.sequential else ""
    tag += f"_lr_{args.lr}" if args.lr != 1e-3 else ""

    num_epochs = {
        "mnist": 15,
        "fmnist": 25,
        "kmnist": 25,
        "emnist": 50,
        "cifar10": 50,
    }[args.dataset]
    if args.sequential:
        num_epochs *= 2

    if args.early_stop:
        num_epochs = 100

    df_all, df_test = pd.DataFrame(), pd.DataFrame()
    
    dirname = pathlib.Path(args.dirname, f"results_{args.dataset}_{args.num_layers}_layer{tag}")

    for model_type, sigma, num_soma, num_dends, t in product(models, sigmas, somata, dendrites, trials):
        fname = dirname / model_type / f"results_sigma_{sigma}_trial_{t}_dends_{num_dends}_soma_{num_soma}.pkl"
        if not fname.exists() or args.force:
            input_args = ""
            if "dropout" in model_type:
                temp = model_type.split('_')
                dropout = float(temp[-1])
                model_index = get_model_idx('_'.join(temp[:-2]))
                input_args += f" --dropout {dropout}"
            else:
                model_index = get_model_idx(model_type)
            
            input_args += f" --trial {t} --model {model_index} --dataset {args.dataset} --num-layers {args.num_layers} --sigma {sigma} -d {num_dends} -s {num_soma} -o {args.dirname} --backend {args.backend}"
            
            print("uv run main.py", *input_args.split())
            print(input_args.split())
            run_model(input_args.split())

        with open(fname, "rb") as f:
            data = pickle.load(f)

        data.pop("masks", None)
        trainable_count = num_trainable_params(D=num_dends, S=num_soma, model_type=model_type)

        meta = {
            "trial": t,
            "model": model_type,
            "sigma": sigma,
            "num_dends": num_dends,
            "num_soma": num_soma,
            "trainable_params": trainable_count,
            "trainable_params_grouped": 10 ** get_power_of_10(trainable_count),
        }

        df = pd.DataFrame(data)
        df["epoch"] = range(1, len(df) + 1)
        for k, v in meta.items():
            df[k] = v

        df_all = df_all._append(df, ignore_index=True)

        df_test_entry = {
            "test_acc": data["test_acc"],
            "test_loss": data["test_loss"],
            "trial": t,
            "model": model_type,
            "sigma": sigma,
            "num_dends": num_dends,
            "num_soma": num_soma,
            "trainable_params": trainable_count,
            "trainable_params_grouped": 10 ** get_power_of_10(trainable_count),
            "num_epochs_min": int(np.argmin(data["val_loss"])),
        }

        df_test = df_test._append(pd.DataFrame([df_test_entry]), ignore_index=True)

    if args.dirname:
        results = {"training": df_all, "testing": df_test}
        
        if args.noise and not args.dropout:
            fname_out = "output_all_noise_final.pkl"
        elif args.dropout and not args.noise:
            fname_out = "output_all_dropout_final.pkl"
        else:
            fname_out = "output_all_final.pkl"

        os.makedirs(dirname, exist_ok=True)
        with open(dirname / fname_out, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Time: {round(time.time() - t1, 2)} sec")

if __name__ == "__main__":
    main()
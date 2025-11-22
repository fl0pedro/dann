#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:00:15 2021.
@author: spiros and assmuth
"""

import os
import argparse
import copy
import json
from optax import adamw
from optax.losses import softmax_cross_entropy_with_integer_labels
import opt_jax
from models import get_model
import jax
import jax.numpy as jnp
import jax.random as jr

def smart_parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="dirname")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-t", "--trial", type=int, default=0)
    parser.add_argument("-d", "--dendrites", "--dends", dest="dends", type=int, nargs="+")
    parser.add_argument("-s", "--soma", type=int, nargs="+")
    parser.add_argument("--gpu", action="store_const", const="1", default="")
    parser.add_argument("--sequential", "--seq", dest="seq", action="store_true")
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--all-to-all", action="store_true")
    parser.add_argument("--conventional", action="store_true")
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--flexi", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--rfs", choices=["somatic", "dendritic"])
    parser.add_argument("--model-id", "--id", type=int)
    parser.add_argument("--model-name", "--name")
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--synapses", dest="nsyns", type=int, default=[16], nargs="+")
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--learning-rate", dest="lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    # depreciated:
    # parser.add_argument("--input-shape", type=int, nargs="+", default=[28, 28, 1])
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--training-split", "--split", type=float, dest="split", default=0.9)
    parser.add_argument("--data-dir")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fmnist", "kmnist", "emnist", "cifar10"],
        default="fmnist",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None):
    config = smart_parse_args(args)

    dataloader, dataset_info = opt_jax.dataloader(config.dataset, config.batch_size, config.data_dir, config.split)
    
    opt_jax.sync_model(config, dataset_info) # config is changed in place.
    unique_model_name = opt_jax.unique_model_name(config)

    key = jr.PRNGKey(config.trial)

    if config.dirname:
        result_files = f"{config.dirname}/*_{unique_model_name}.pkl"

        if os.path.isfile(result_files) and not config.force:
            raise FileExistsError(f"This run has already been recorded: {result_files}")

    key, model, *params = get_model(key, config)

    optimizer = adamw(config.lr)
    loss_fn = softmax_cross_entropy_with_integer_labels

    untrained_params = copy.deepcopy(params)
    
    params, res = opt_jax.train_loop(key,
        model, params, dataloader, loss_fn, optimizer,
        batch_size=config.batch_size, epochs=config.epochs,
        shuffle=not config.seq, early_stop=config.early_stop,
    )

    if config.dirname:
        os.makedirs(config.dirname, exist_ok=True)
        with open(f"{config.dirname}/untrained_{unique_model_name}.pkl", "wb") as f:
            for x in jax.tree.leaves(untrained_params):
                jnp.save(f, x)
        with open(f"trained_{unique_model_name}.pkl", "wb") as f:
            for x in jax.tree.leaves(params):
                jnp.save(f, x)
        with open(f"result_{unique_model_name}.pkl", "wb") as f:
            json.dump(res, f)
        print(f"\nResults saved: {config.dirname}/*_{unique_model_name}.pkl")

    return unique_model_name, res

if __name__ == "__main__":
    main()

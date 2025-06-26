#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:00:15 2021.
@author: spiros and assmuth
"""
import os
import argparse
import pathlib
import pickle
import copy
from functools import cache, partial

def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--num-dendrites", type=int, required=True)
    parser.add_argument("-s", "--num-somas", type=int, required=True)
    parser.add_argument("-o", "--output", dest="dirname")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--gpu", action="store_const", const="1", default="")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--all-to-all", action="store_true")
    parser.add_argument("--conventional", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--rfs", choices=["somatic", "dendritic"])
    parser.add_argument("--model", type=int)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-synapses", dest="nsyns", type=int, default=16)
    parser.add_argument("--drop-rate", type=float, default=0)
    parser.add_argument("--learning-rate", dest="lr", type=float, default=1e-3)
    parser.add_argument("--dataset", choices=["mnist", "fmnist", "kmnist", "emnist", "cifar10"], default="fmnist")
    parser.add_argument("--backend", choices=["tensorflow", "torch", "jax"], default="jax")
    return parser.parse_args(args)

from collections import namedtuple

@cache
def init(backend: str, gpu: str):
    from opt import (
        custom_train_loop_tensorflow, custom_train_loop_torch, custom_train_loop_jax,
        make_masks, get_model, get_data, get_model_name, init_keras
    )
    keras, _ = init_keras(backend, gpu)

    BackendContext = namedtuple("BackendContext", [
        "keras", "train_loops", "make_masks", "get_model", "get_data", "get_model_name"
    ])
    device = "gpu" if gpu == "1" else "cpu"
    return BackendContext(
        keras=keras,
        train_loops={
            "tensorflow": partial(custom_train_loop_tensorflow, device=device),
            "torch": partial(custom_train_loop_torch, device=device),
            "jax": partial(custom_train_loop_jax, device=device),
        },
        make_masks=make_masks,
        get_model=partial(get_model, backend=backend, device=device),
        get_data=partial(get_data, backend=backend, device=device),
        get_model_name=get_model_name
    )


def main(args: list[str] | None = None):
    args = parse_args(args)
    # print(args)

    backend = init(args.backend, args.gpu)

    run_tag = ""
    if args.sequential:
        run_tag += "_sequential"
    if args.lr != 1e-3:
        run_tag += f"_lr_{args.lr}"

    fname_model = backend.get_model_name(
        args.conventional,
        args.rfs is not None,
        args.sparse,
        args.rfs,
        "all_to_all" if args.all_to_all else None
    )
    if args.drop_rate:
        fname_model += f"_dropout_{args.drop_rate}"

    print(f"\nModel: {fname_model}, trial: {args.trial}, layers: {args.num_layers}, "
          f"noise: {args.sigma}, dataset: {args.dataset}, tag: {run_tag}\n")

    if args.dirname:
        subdirname = f"results_{args.dataset}_{args.num_layers}_layer{run_tag}"
        fulldir = pathlib.Path(args.dirname, subdirname, fname_model)

        postfix = f"sigma_{args.sigma}_trial_{args.trial}_dends_{args.num_dendrites}_soma_{args.num_somas}"
        result_file = fulldir / f"results_{postfix}.pkl"

        if result_file.exists() and not args.force:
            print(f"This run has already been recorded: {result_file}")
            return

    backend.keras.utils.set_random_seed(args.trial)

    update_model_config(args)

    batch_size = 128
    data, labels, img_height, img_width, channels = backend.get_data(
        validation_split=0.1,
        dtype=args.dataset,
        normalize=True,
        add_noise=bool(args.sigma),
        sigma=args.sigma,
        sequential=args.sequential,
        batch_size=batch_size,
        seed=args.trial,
    )

    x_train, x_val, x_test = data["train"], data["val"], data["test"]
    y_train, y_val, y_test = labels["train"], labels["val"], labels["test"]

    num_classes = len(set(y_train))
    dends = args.num_layers * [args.num_dendrites]
    soma = args.num_layers * [args.num_somas]

    input_shape = (img_width * img_height * channels,)
    model = backend.get_model(
        input_shape, args.num_layers, dends, soma, num_classes,
        fname_model=fname_model, dropout=bool(args.drop_rate), rate=args.drop_rate,
    )

    masks = backend.make_masks(
        dends, soma, args.nsyns, args.num_layers,
        img_width, img_height, num_classes, channels,
        conventional=args.conventional,
        rfs=args.rfs is not None,
        rfs_type=args.rfs,
        rfs_mode="random",
        seed=args.trial,
    )

    params = model.get_weights()
    model.set_weights([p * m for p, m in zip(params, masks)])
    model_untrained = copy.deepcopy(model)

    optimizer = backend.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = backend.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

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

    train_fn = backend.train_loops[args.backend]

    model, out = train_fn(
        model, loss_fn, optimizer, masks, batch_size, num_epochs,
        x_train, y_train, x_val, y_val, x_test, y_test,
        shuffle=not args.sequential, early_stop=args.early_stop, patience=10,
    )

    out["masks"] = masks

    if args.dirname:
        os.makedirs(fulldir, exist_ok=True)
        model_untrained.save(fulldir / f"untrained_model_{postfix}.keras")
        model.save(fulldir / f"model_{postfix}.keras")
        with open(result_file, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nResults saved: {fulldir}/[untrained_]model_{postfix}.keras+pkl")

def update_model_config(args):
    match args.model:
        case 0:
            pass
        case 1:
            args.rfs = "somatic"
        case 2:
            args.rfs = "dendritic"
        case 3:
            args.conventional = True
        case 4:
            args.conventional = True
            args.sparse = True
        case 5:
            args.conventional = True
            args.rfs = "somatic"
        case 6:
            args.conventional = True
            args.rfs = "dendritic"
        case 7:
            args.sparse = True
        case 8:
            args.rfs = "somatic"
            args.sparse = True
        case 9:
            args.rfs = "dendritic"
            args.sparse = True
        case 10:
            args.rfs = "somatic"
            args.all_to_all = True
        case 11:
            args.rfs = "somatic"
            args.sparse = True
            args.all_to_all = True

if __name__ == "__main__":
    main()
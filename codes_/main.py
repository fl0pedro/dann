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
from collections import namedtuple
import numpy as np


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
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fmnist", "kmnist", "emnist", "cifar10"],
        default="fmnist",
    )
    parser.add_argument(
        "--backend", choices=["tensorflow", "torch", "jax", "tiny"], default="jax"
    )
    parser.add_argument("--mask", action="store_true")
    return parser.parse_args(args)


@cache
def init(backend: str, gpu: str):
    from opt import make_masks, get_data, get_model_name, init_keras

    keras, _ = init_keras(backend, gpu)

    BackendContext = namedtuple(
        "BackendContext",
        [
            "keras",
            "train_loop",
            "make_masks",
            "dann",
            "mlc",
            "get_data",
            "get_model_name",
        ],
    )

    if backend == "tiny":
        from tiny_backend import train_loop, TinyDANN, MultiLayerLocallyConnected2D

        return BackendContext(
            keras=None,
            train_loop=train_loop,
            make_masks=make_masks,
            dann=TinyDANN,
            mlc=MultiLayerLocallyConnected2D,
            get_data=get_data,
            get_model_name=get_model_name,
        )
    else:
        from opt import (
            custom_train_loop_tensorflow,
            custom_train_loop_torch,
            custom_train_loop_jax,
            make_masks,
            get_model,
            get_model_name,
            init_keras,
        )
        from LocallyConnected2d import MultiLayerLocallyConnected2D

        device = "gpu" if gpu == "1" else "cpu"

        train_loop = {
            "tensorflow": custom_train_loop_tensorflow,
            "torch": custom_train_loop_torch,
            "jax": custom_train_loop_jax,
        }[backend]
        train_loop = partial(train_loop, device=device)

        return BackendContext(
            keras=keras,
            train_loop=train_loop,
            make_masks=make_masks,
            dann=partial(get_model, backend=backend, device=device),
            mlc=MultiLayerLocallyConnected2D,
            get_data=partial(get_data, backend=backend, device=device),
            get_model_name=get_model_name,
        )


def main(args: list[str] | None = None):
    args = parse_args(args)
    # print(args)

    # if args.model == 12:
    #     args.backend = "jax"

    backend = init(args.backend, args.gpu)

    run_tag = ""
    if args.sequential:
        run_tag += "_sequential"
    if args.lr != 1e-3:
        run_tag += f"_lr_{args.lr}"

    fname_model = backend.get_model_name(idx=args.model)
    if args.drop_rate:
        fname_model += f"_dropout_{args.drop_rate}"

    print(
        f"\nModel: {fname_model}, trial: {args.trial}, layers: {args.num_layers}, "
        f"noise: {args.sigma}, dataset: {args.dataset}, tag: {run_tag}\n"
    )

    if args.dirname:
        subdirname = f"results_{args.dataset}_{args.num_layers}_layer{run_tag}"
        fulldir = pathlib.Path(args.dirname, subdirname, fname_model)

        postfix = f"sigma_{args.sigma}_trial_{args.trial}_dends_{args.num_dendrites}_soma_{args.num_somas}"
        result_file = fulldir / f"results_{postfix}.pkl"

        if result_file.exists() and not args.force:
            print(f"This run has already been recorded: {result_file}")
            return

    if backend.keras is not None:
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

    # if args.model == 12:
    #     rs = lambda x: x.reshape(-1, img_height, img_width, channels)
    #     data = {
    #         k: rs(v) if not isinstance(v, list) else [rs(x) for x in v]
    #         for k, v in data.items()
    #     }

    x_train, x_val, x_test = data["train"], data["val"], data["test"]
    y_train, y_val, y_test = labels["train"], labels["val"], labels["test"]

    dends = args.num_layers * [args.num_dendrites]
    soma = args.num_layers * [args.num_somas]

    num_classes, num_epochs = {
        "mnist": (10, 15),
        "fmnist": (10, 25),
        "kmnist": (10, 25),
        "emnist": (47, 50),  # may be wrong*
        "cifar10": (10, 50),
    }[args.dataset]
    if args.sequential:
        num_epochs *= 2
    if args.early_stop:
        num_epochs = 100

    if args.model == 12:
        kernels = [
            (int(np.sqrt(k)) + 1,) * 2 for pairs in zip(dends, soma) for k in pairs
        ]
        strides = [
            tuple(x // int(np.sqrt(args.nsyns)) or 1 for x in k) for k in kernels
        ]
        model = backend.mlc(
            input_shape=(img_width, img_height),
            layer_depth=[channels, 1],
            output_size=10,
            kernels=kernels,
            strides=strides,
            bias=True,
        )
        masks = None
    else:
        input_shape = (img_width * img_height * channels,)

        masks = backend.make_masks(
            dends,
            soma,
            args.nsyns,
            args.num_layers,
            img_width,
            img_height,
            num_classes,
            channels,
            conventional=args.conventional,
            rfs=args.rfs is not None,
            rfs_type=args.rfs,
            rfs_mode="random",
            seed=args.trial,
        )

        model = backend.dann(
            input_shape,
            args.num_layers,
            dends,
            soma,
            num_classes,
            fname_model=fname_model,
            dropout=bool(args.drop_rate),
            rate=args.drop_rate,
            masks=masks if args.mask else None
        )

        print((
            dends,
            soma,
            args.nsyns,
            args.num_layers,
            img_width,
            img_height,
            num_classes,
            channels,
            args.conventional,
            args.rfs is not None,
            args.rfs,
            "random",
            args.trial,
        ))



        # params = model.get_weights()
        # model.set_weights([p * m for p, m in zip(params, masks)])

        # print(*[f"{l.name}:\n\tweights: {l.weights[0].shape}\n\tbias: {l.weights[1].shape}" for l in model.layers if len(l.weights) > 0], sep="\n\n")
        # model.params = params

    if args.backend == "tiny":
        from tinygrad import Tensor
        x_train, x_val, x_test = [
            Tensor(x).reshape(x.shape[0], img_width, img_height, channels) for x in 
            [x_train, x_val, x_test]
        ]
 
        y_train, y_val, y_test = [
            Tensor(x) for x in 
            [y_train, y_val, y_test]
        ]
        
        masks = [Tensor(m) for m in masks] if masks is not None else None
        
        
        import cProfile
        import pstats
        import io
        
        pr = cProfile.Profile()
        pr.enable()
        backend.train_loop(
            model,
            masks,
            batch_size,
            num_epochs,
            x_train,
            x_val,
            x_test,
            y_train,
            y_val,
            y_test,
            args.lr,
            not args.sequential,
        )

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
        ps.print_stats(20)
        print(s.getvalue())

        return

    optimizer = backend.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = backend.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    print("deepcopy")
    model_untrained = copy.deepcopy(model)
    
    print("train")
    model, out = backend.train_loop(
        model,
        loss_fn,
        optimizer,
        masks,
        batch_size,
        num_epochs,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        shuffle=not args.sequential,
        early_stop=args.early_stop,
        patience=10,
    )

    print(model.summary())

    if args.model != 12:
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

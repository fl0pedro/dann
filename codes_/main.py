#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:00:15 2021.

@author: spiros and florian
"""
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--num-dendrites", type=int, required=True)
parser.add_argument("-s", "--num-somas", type=int, required=True)
parser.add_argument("-o", "--output", dest="dirname")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--force", action="store_true")
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

args = parser.parse_args()

print(args)

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))

import copy
import keras
import pickle
import pathlib

from opt import make_masks, custom_train_loop, get_model, get_data, get_model_name

run_tag = ""
if args.sequential:
    run_tag += "_sequential"
if args.lr != 1e-3: # i.e., not default
    run_tag += f"_lr_{args.lr}"

fname_model = get_model_name(
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

    if fulldir.exists() and not args.force:
        print("This run has already been recorded")
        exit(0)


keras.utils.set_random_seed(args.trial)

match args.model:
    case 0: # dendritic ANN (dANN) with random connections
        pass
    case 1: # dendritic ANN (dANN) with global RFs
        args.rfs = "somatic"
    case 2: # dendritic ANN (dANN) with local RFs
        args.rfs = "dendritic"
    case 3: # vanilla ANN (vANN) all-to-all connections (no RFs)
        args.conventional = True
    case 4: # vanilla ANN (vANN) with random connections
        args.conventional = True
        args.sparse = True
    case 5: # vanilla ANN (vANN) with global RFs
        args.conventional = True
        args.rfs = "somatic"
    case 6: # vanilla ANN (vANN) with local RFs
        args.conventional = True
        args.rfs = "dendritic"
    case 7: # sparse ANN (sANN)
        args.sparse = True
    case 8: # sparse ANN (sANN) with global RFs
        args.rfs = "somatic"
        args.sparse = True
    case 9: # sparse ANN (sANN) with local RFs
        args.rfs = "dendritic"
        args.sparse = True
    case 10: # dendritic ANN (dANN) with all-to-all inputs
        args.rfs = "somatic"
        args.all_to_all = True
    case 11: # sparse ANN (sANN) with all-to-all inputs
        args.conventional = False
        args.rfs = "somatic"
        args.sparse = True
        args.all_to_all = True

batch_size = 128
data, labels, img_height, img_width, channels = get_data(
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
dends = args.num_layers*[args.num_dendrites]
soma = args.num_layers*[args.num_somas]


input_shape = (img_width * img_height * channels, )
model = get_model(
    input_shape,
    args.num_layers,
    dends,
    soma,
    num_classes,
    fname_model=fname_model,
    dropout=bool(args.drop_rate),
    rate=args.drop_rate,
)


masks = make_masks(
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

params = model.get_weights()
params_masked = [params[i]*masks[i] for i in range(len(params))]

model.set_weights(params_masked)
model_untrained = copy.deepcopy(model)

# Instantiate the optimizer and the loss function
optimizer = keras.optimizers.Adam(learning_rate=args.lr)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

match args.dataset:
    case "mnist":
        num_epochs = 15
    case "fmnist":
        num_epochs = 25
    case "kmnist":
        num_epochs = 25
    case "emnist":
        num_epochs = 50
    case "cifar10":
        num_epochs = 50

if args.sequential:
    num_epochs *= 2

if args.early_stop:
    num_epochs = 100


model, out = custom_train_loop(
    model, loss_fn, optimizer,
    masks, batch_size,
    num_epochs,
    x_train, y_train,
    x_val, y_val,
    x_test, y_test,
    shuffle=not args.sequential,
    early_stop=args.early_stop,
    patience=10,
)


out["masks"] = masks

if args.dirname:
    os.makedirs(fulldir, exist_ok=True)
    
    model_untrained.save(fulldir / f"untrained_model_{postfix}.keras")
    model.save(fulldir / f"model_{postfix}.keras")

    with open(fulldir / f"results_{postfix}.pkl", "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nResults have been saved in: {fulldir}/[untrained_]model_{postfix}.keras+pkl")

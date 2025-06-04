#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:00:15 2021.

@author: spiros
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import sys
import copy
import keras
import pickle
import pathlib

from opt import make_masks
from opt import custom_train_loop
from opt import get_model
from opt import get_data
from opt import get_model_name

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--num-dendrites", type=int, required=True)
parser.add_argument("-s", "--num-somas", type=int, required=True)
parser.add_argument("-o" "--output", action="store_true")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--sequential", action="store_true")
parser.add_argument("--early-stop", action="store_true")
parser.add_argument("--all-to-all", action="store_true")
parser.add_argument("--conventional", action="store_true")
parser.add_argument("--sparse", action="store_true")
parser.add_argument("--rfs", choices=["somatic", "dendritic"])
parser.add_argument("--model", type=int) # deprecated
parser.add_argument("--trial", type=int, default=0)
parser.add_argument("--sigma", type=float, default=0.0)
parser.add_argument("--num-layers", type=int, default=1)
parser.add_argument("--num-synapses", dest="nsyn", type=int, default=16)
parser.add_argument("--drop-rate", type=float, default=0)
parser.add_argument("--learning-rate", dest="lr", type=float, default=1e-3)
parser.add_argument("--dataset", choices=["mnist", "fmnist", "kmnist", "emnist", "cifar10"], default="fmnist")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))

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

# Get the data
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

# Extract the data in train, validation, and test sets
x_train, x_val, x_test = data["train"], data["val"], data["test"]
y_train, y_val, y_test = labels["train"], labels["val"], labels["test"]

# Model architectures
num_classes = len(set(y_train))
dends = args.num_layers*[args.num_dendrites]
soma = args.num_layers*[args.num_soma]

# Build the masks
Masks = make_masks(
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

# Get the model
input_shape = (img_width * img_height * channels, )

fname_model = get_model_name(
    args.conventional,
    args.rfs is not None,
    args.sparse,
    args.rfs,
    "all_to_all" if args.all_to_all else None
)

# Set the foldername extension
file_tag = ""
if args.sequential:
    file_tag += "_sequential"

# Change the model name if dropout
if args.drop_rate:
    fname_model += f"_dropout_{args.drop_rate}"

# Get the model
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

# Apply the masks to initial weights
PARAMS = model.get_weights()
PARAMSmod = [PARAMS[i]*Masks[i] for i in range(len(PARAMS))]

# Set the initial weights by zeroing out not connected nodes.
model.set_weights(PARAMSmod)
model_untrained = copy.deepcopy(model)

# Instantiate the optimizer and the loss function
lr = float(sys.argv[14])
optimizer = keras.optimizers.Adam(learning_rate=lr)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

if lr != 1e-3: # i.e., not default
    file_tag += f"_lr_{lr}"

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

print(f"\nModel: {fname_model}, trial: {args.trial}, layers: {args.num_layers}, "
      f"noise: {args.sigma}, dataset: {args.dataset}, tag: {file_tag}\n")

# train and evaluate the model
model, out = custom_train_loop(
    model, loss_fn, optimizer,
    Masks, batch_size,
    num_epochs,
    x_train, y_train,
    x_val, y_val,
    x_test, y_test,
    shuffle=not args.sequential,
    early_stop=args.early_stop,
    patience=10,
)

# Store masks in the output dictionary
out["masks"] = Masks

if args.output:
    # the local directory to save the data
    path_to_dir_local = sys.argv[15]
    if not os.path.exists(path_to_dir_local):
        os.mkdir(path_to_dir_local)

    # subdirectory with name of model, num of layers and other tags added
    sub_tag = f"results_{args.dataset}_{args.num_layers}_layer{file_tag}/"
    dirname = pathlib.Path(f"{path_to_dir_local}/{sub_tag}")
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    # Create the output directory
    outdir_name = pathlib.Path(f"{dirname}/{fname_model}")
    if not os.path.exists(outdir_name):
        os.mkdir(outdir_name)

    # Save the model
    postfix = f"sigma_{args.sigma}_trial_{args.trial}_dends_{args.num_dends}_soma_{args.num_soma}"
    print(f"Saving to: {outdir_name}/model_{postfix}.keras+pkl")
    # Save the untrained and trained model
    # model_untrained.save(pathlib.Path(f"{outdir_name}/untrained_model_{postfix}.h5"))
    model.save(pathlib.Path(f"{outdir_name}/model_{postfix}.keras"))

    # Save the results
    fname_res = pathlib.Path(f"{outdir_name}/results_{postfix}.pkl")
    with open(fname_res, "wb") as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nResults have been saved in: {dirname}/{fname_model}")

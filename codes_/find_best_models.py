#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:45:25 2024

@author: spiros
"""
import os
import pickle
import pathlib
import pandas as pd

from plotting_functions import find_best_models, model_config
from plotting_functions import fix_names, keep_models

# data store directory
data_dir = "../DATA/"

dirname_figs = '../FinalFigs_manuscript'
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

seq = False
seq_tag = "_sequential" if seq else ""
datatypes = ["mnist", "fmnist", "kmnist", "emnist", "cifar10"]
num_layers = 1

df_best_acc = pd.DataFrame()
df_best_loss = pd.DataFrame()
df_high_acc = pd.DataFrame()
df_low_loss = pd.DataFrame()
for datatype in datatypes:

    dirname = f"{data_dir}/results_{datatype}_{num_layers}_layer/"

    # Load data (deserialize)
    fname_store =  pathlib.Path(f"{dirname}/output_all_final")
    with open(f'{fname_store}.pkl', 'rb') as file:
        results = pickle.load(file)
    df_all = fix_names(results['training'])
    df_test = fix_names(results['testing'])

    # Keep models to plot, i.e., dend ANN and vanilla ANN.
    model_to_keep = [
        'dANN-R', 'dANN-LRF', 'dANN-GRF', 'pdANN',
        'sANN', 'sANN-LRF', 'sANN-GRF','psANN',
        'vANN-R', 'vANN-LRF', 'vANN-GRF',
        'vANN',
    ]

    df_all_ = keep_models(df_all, model_to_keep)
    df_test_ = keep_models(df_test, model_to_keep)

    df_all_['train_acc'] *= 100
    df_all_['val_acc'] *= 100
    df_test_['test_acc'] *= 100

    # Find the best models for each model_type
    models_high_accuracy_all = find_best_models(
        df_test_,
        model_to_keep,
        metric='accuracy',
        compare=True
        )
    models_low_loss_all = find_best_models(
        df_test_,
        model_to_keep,
        metric='loss',
        compare=True
        )
    models_best_accuracy_all = find_best_models(
        df_test_,
        model_to_keep,
        metric='accuracy',
        compare=False
        )
    models_best_loss_all = find_best_models(
        df_test_,
        model_to_keep,
        metric='loss',
        compare=False
        )

    # Load data (deserialize)
    fname_store = pathlib.Path(f"{dirname}/output_all_final")
    with open(f'{fname_store}.pkl', 'rb') as file:
        results = pickle.load(file)
    df_test = fix_names(results['testing'])

    # Keep models to plot, i.e., dend ANN and vanilla ANN.
    model_to_keep = [
        'dANN-R', 'dANN-LRF', 'dANN-GRF', 'pdANN',
        'sANN', 'sANN-LRF', 'sANN-GRF','psANN',
        'vANN-R', 'vANN-LRF', 'vANN-GRF',
        'vANN',
    ]

    df_test_ = keep_models(df_test, model_to_keep)
    df_test_['test_acc'] *= 100

    for model_type in model_to_keep:
        config_ = models_best_accuracy_all[model_type]
        df_ = model_config(
            df_test_,
            d=int(config_[0]),
            s=int(config_[1]),
            m=model_type)
        df_['data'] = datatype
        df_best_acc = pd.concat([df_best_acc, df_])

        config_ = models_best_loss_all[model_type]
        df_ = model_config(
            df_test_,
            d=int(config_[0]),
            s=int(config_[1]),
            m=model_type)
        df_['data'] = datatype
        df_best_loss = pd.concat([df_best_loss, df_])

        config_ = models_high_accuracy_all[model_type]
        df_ = model_config(
            df_test_,
            d=int(config_[0]),
            s=int(config_[1]),
            m=model_type)
        df_['data'] = datatype
        df_high_acc = pd.concat([df_high_acc, df_])

        config_ = models_low_loss_all[model_type]
        df_ = model_config(
            df_test_,
            d=int(config_[0]),
            s=int(config_[1]),
            m=model_type)
        df_['data'] = datatype
        df_low_loss = pd.concat([df_low_loss, df_])


result = {}
result['compare_acc'] = df_high_acc
result['compare_loss'] = df_low_loss
result['best_acc'] = df_best_acc
result['best_loss'] = df_best_loss

# Save the results
fname = f"{data_dir}/all_datasets_best_models{seq_tag}_final.pkl"
with open(fname, 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

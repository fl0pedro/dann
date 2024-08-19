#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:24:53 2023

@author: spiros
"""
import os
import sys
import time
import pickle
import pathlib
import numpy as np
import pandas as pd

from utils import num_trainable_params
from utils import get_power_of_10

t1 = time.time()

# Simulation parameters
dir_full_path = sys.argv[1]
datatype = sys.argv[2]  # datatype (mnist, fmnist, kmnist, emnist, cifar10)
save_data = True
seq = int(sys.argv[3])
noise = int(sys.argv[4])
drop = int(sys.argv[5])
num_layers = int(sys.argv[6])
lr = float(sys.argv[7])
early_stop_flag = int(sys.argv[8])

if drop == 0:
    models = [
        'dend_ann_random',
        'dend_ann_local_rfs',
        'dend_ann_global_rfs',
        'dend_ann_all_to_all',
        'vanilla_ann',
        'vanilla_ann_local_rfs',
        'vanilla_ann_global_rfs',
        'vanilla_ann_random',
        'sparse_ann',
        'sparse_ann_local_rfs',
        'sparse_ann_global_rfs',
        'sparse_ann_all_to_all',
    ]
elif drop == 1:
    models = [
        'dend_ann_random',
        'dend_ann_local_rfs',
        'dend_ann_global_rfs',
        'vanilla_ann',
        'vanilla_ann_dropout_0.2',
        'vanilla_ann_dropout_0.5',
        'vanilla_ann_dropout_0.8',
    ]

trials = 5

somata = [2**i for i in range(5, 10)]  # number of somata (i.e., somatic, second layer)
dendrites = [2**i for i in range(7)]  # number of dendrites per soma (i.e., dendritic layer)

tag = "" if seq == 0 else "_sequential"
tag2 = "" if lr == 1e-3 else f"_lr_{lr}"
sigmas = [0.0] if noise == 0 else [0.0, 0.25, 0.5, 0.75, 1.0]

# Hyperparameters
if datatype == 'mnist':
    n_epochs = 15 if not seq else 30
elif datatype == 'fmnist' or datatype == 'kmnist':
    n_epochs = 25 if not seq else 50
elif datatype == 'cifar10' or datatype == 'emnist':
    n_epochs = 50

if early_stop_flag == 1:
    n_epochs = 100

# data frame to store all analysis for training, validation and test
print(f"\nresults_{datatype}_{num_layers}_layer{tag}{tag2}/")

df_all = None
df_test = None
for model_type in models:
    for sigma in sigmas:
        for num_soma in somata:
            for num_dends in dendrites:
                for t in range(1, trials+1):
                    print(f"Analyze model: {model_type} with {num_soma} somata"
                          f" and {num_dends} dendrites for trials {t}, sigma {sigma}")
                    # Set up directory and file names
                    dirname = f"{dir_full_path}/results_{datatype}_{num_layers}_layer{tag}{tag2}/"
                    subdir = f"{model_type}"
                    postfix = f"sigma_{sigma}_trial_{t}_dends_{num_dends}_soma_{num_soma}"
                    fname_data = pathlib.Path(f"{dirname}/{subdir}/results_{postfix}.pkl")

                    # load the training/validation results
                    if not os.path.isfile(fname_data):
                        continue

                    with open(fname_data, 'rb') as file:
                        DATA = pickle.load(file)

                    trainable_count = num_trainable_params(
                        D=num_dends,
                        S=num_soma,
                        model_type=model_type
                    )
                    # remove the key 'Masks'
                    if 'Masks' in DATA.keys():
                        DATA.pop('Masks')
                    # Create the DataFrame
                    if df_all is None:
                        df_all = pd.DataFrame.from_dict(DATA)
                        df_all['trial'] = t
                        df_all['model'] = model_type
                        df_all['epoch'] = range(1, len(df_all)+1)
                        df_all['sigma'] = sigma
                        df_all['num_dends'] = num_dends
                        df_all['num_soma'] = num_soma
                        df_all['trainable_params'] = trainable_count
                        df_all['trainable_params_grouped'] = 10**get_power_of_10(trainable_count)
                    else:
                        df_ = pd.DataFrame.from_dict(DATA)
                        df_['trial'] = t
                        df_['model'] = model_type
                        df_['epoch'] = range(1, len(df_)+1)
                        df_['sigma'] = sigma
                        df_['num_dends'] = num_dends
                        df_['num_soma'] = num_soma
                        df_['trainable_params'] = trainable_count
                        df_['trainable_params_grouped'] = 10**get_power_of_10(trainable_count)
                        df_all = pd.concat([df_all, df_], ignore_index=True)

                    # Create the DataFrame
                    if df_test is None:
                        df_test = pd.DataFrame(
                            columns=['test_acc',
                                     'test_loss',
                                     'model',
                                     'num_dends',
                                     'num_soma',
                                     'sigma',
                                     'trial'],
                            index=[0]
                            )
                        df_test['test_acc'] = DATA['test_acc']
                        df_test['test_loss'] = DATA['test_loss']
                        df_test['trial'] = t
                        df_test['model'] = model_type
                        df_test['sigma'] = sigma
                        df_test['num_dends'] = num_dends
                        df_test['num_soma'] = num_soma
                        df_test['trainable_params'] = trainable_count
                        df_test['trainable_params_grouped'] = 10**get_power_of_10(trainable_count)
                        df_test['num_epochs_min'] = np.argmin(DATA['val_loss'])
                    else:
                        df_ = pd.DataFrame(
                            columns=['test_acc',
                                     'test_loss',
                                     'model',
                                     'num_dends',
                                     'num_soma',
                                     'sigma',
                                     'trial'],
                            index=[0]
                            )
                        df_['test_acc'] = DATA['test_acc']
                        df_['test_loss'] = DATA['test_loss']
                        df_['trial'] = t
                        df_['model'] = model_type
                        df_['sigma'] = sigma
                        df_['num_dends'] = num_dends
                        df_['num_soma'] = num_soma
                        df_['trainable_params'] = trainable_count
                        df_['trainable_params_grouped'] = 10**get_power_of_10(trainable_count)
                        df_['num_epochs_min'] = np.argmin(DATA['val_loss'])
                        df_test = pd.concat([df_test, df_], ignore_index=True)

if save_data:
    results = {}
    results['training'] = df_all
    results['testing'] = df_test
    # Store data (serialize)
    if noise and not drop:
        fname_store = pathlib.Path(f"{dirname}/output_all_noise_final.pkl")
    elif drop and not noise:
        fname_store = pathlib.Path(f"{dirname}/output_all_dropout_final.pkl")
    else:
        fname_store = pathlib.Path(f"{dirname}/output_all_final.pkl")

    with open(f'{fname_store}', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Time taken to analyze and save data: {round(time.time() - t1, 2)} sec")

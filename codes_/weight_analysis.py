#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:52:39 2023

@author: spiros
"""

import os
import keras
import pickle
import pathlib
import numpy as np
import pandas as pd

from utils import remove_zeros, get_layer_weights
from plotting_functions import find_best_models, model_config
from plotting_functions import fix_names, keep_models


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data_dir = "../DATA/"

dirname_figs = '../FinalFigs_manuscript'
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

datatype = "fmnist"
seq = False
seq_tag = "_sequential" if seq else ""
num_layers = 1

dirname = f"{data_dir}/results_{datatype}_{num_layers}_layer{seq_tag}/"

# Load data (deserialize)
fname_store = pathlib.Path(f"{dirname}/output_all_new")
with open(f'{fname_store}.pkl', 'rb') as file:
    results = pickle.load(file)
df_all = fix_names(results['training'])
df_test = fix_names(results['testing'])

# Keep models to plot, i.e., dend ANN and vanilla ANN.
model_to_keep = [
    'dANN-R', 'dANN-LRF', 'dANN-GRF', 'dANN-F',
    'sANN', 'sANN-LRF', 'sANN-GRF', 'sANN-F',
    'vANN-R', 'vANN-LRF', 'vANN-GRF',
    'vANN',
]

df_all_ = keep_models(df_all, model_to_keep)
df_test_ = keep_models(df_test, model_to_keep)
df_all_['train_acc'] *= 100
df_all_['val_acc'] *= 100
df_test_['test_acc'] *= 100

# Find the best models for each model_type
models_best = find_best_models(df_test_, model_to_keep, metric='accuracy')

# Analyze the weights
sigma = 0.0
n_trials = 5
num_classes = 10

w_soma_all = {}
w_dend_all = {}
w_out_all = {}

models = [
    'dend_ann_random', 'dend_ann_local_rfs',
    'dend_ann_global_rfs', 'dend_ann_all_to_all',
    'sparse_ann', 'sparse_ann_global_rfs',
    'sparse_ann_local_rfs', 'sparse_ann_all_to_all',
    'vanilla_ann_random', 'vanilla_ann_local_rfs',
    'vanilla_ann_global_rfs', 'vanilla_ann',
]

for model_, model_type in zip(model_to_keep, models):
    num_soma = int(models_best[model_][1])
    num_dends = int(models_best[model_][0])
    num_syns = 28*28 if model_type == 'vanilla_ann' else 16
    K = num_dends*num_soma**2 if 'vanilla_ann' in model_type else num_dends*num_soma
    w_soma_ = np.zeros((K, n_trials))
    w_dend_ = np.zeros((num_syns*num_dends*num_soma, n_trials))
    w_out_ = np.zeros((num_classes*num_soma, n_trials))
    for t in range(1, n_trials+1):
        postfix = f"sigma_{sigma}_trial_{t}_dends_{num_dends}_soma_{num_soma}"
        # Paths to model files
        fname_model = pathlib.Path(f"{dirname}/{model_type}/model_{postfix}.keras")
        # load the trained model
        model = keras.models.load_model(fname_model)
        # load the weights of dendritic and somatic layer
        wd = remove_zeros(get_layer_weights(model, layer_name='dend_1'))
        ws = remove_zeros(get_layer_weights(model, layer_name='soma_1'))
        wo = remove_zeros(get_layer_weights(model, layer_name='output'))
        if wd.shape[0] != w_dend_.shape[0]:
            diff = w_dend_.shape[0] - wd.shape[0]
            wd = np.concatenate((wd, np.zeros((diff, ))))
        w_dend_[:, t-1] = wd
        w_soma_[:, t-1] = ws
        w_out_[:, t-1] = wo

    w_soma_all[f'{model_type}'] = w_soma_
    w_dend_all[f'{model_type}'] = w_dend_
    w_out_all[f'{model_type}'] = w_out_

weights_ = {}
weights_['soma'] = w_soma_all
weights_['dendrites'] = w_dend_all
weights_['output'] = w_out_all

# Save the results
fname = f"{data_dir}/weights_best_models{seq_tag}.pkl"
with open(pathlib.Path(fname), 'wb') as handle:
    pickle.dump(weights_, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Make entropy histograms
fname = "output_all_post_analysis_all_new.pkl"
with open(pathlib.Path(f'{dirname}{fname}'), 'rb') as file:
    DATA = pickle.load(file)

H_dends_all = {}
H_soma_all = {}
for model_, model_type in zip(model_to_keep, models):
    H_dends, H_soma = [], []
    num_soma = int(models_best[model_][1])
    num_dends = int(models_best[model_][0])
    for t in range(1, n_trials + 1):
        H_dends += list(DATA['entropy'][model_type][f'd_{num_dends}_s_{num_soma}'][f'trial_{t}'][0])
        H_soma += list(DATA['entropy'][model_type][f'd_{num_dends}_s_{num_soma}'][f'trial_{t}'][1])

    H_dends_all[model_type] = H_dends
    H_soma_all[model_type] = H_soma

entropies_ = {}
entropies_['soma'] = H_soma_all
entropies_['dendrites'] = H_dends_all
# Save the results
fname = f"{data_dir}/entropies_best_models{seq_tag}.pkl"
with open(pathlib.Path(fname), 'wb') as handle:
    pickle.dump(entropies_, handle, protocol=pickle.HIGHEST_PROTOCOL)

sel_dends_all = {}
sel_soma_all = {}
for model_, model_type in zip(model_to_keep, models):
    sel_dends, sel_soma = [], []
    num_soma = int(models_best[model_][1])
    num_dends = int(models_best[model_][0])
    for t in range(1, n_trials + 1):
        sel_dends += list(DATA['selectivity'][model_type][f'd_{num_dends}_s_{num_soma}'][f'trial_{t}'][0])
        sel_soma += list(DATA['selectivity'][model_type][f'd_{num_dends}_s_{num_soma}'][f'trial_{t}'][1])

    sel_dends_all[model_type] = sel_dends
    sel_soma_all[model_type] = sel_soma

selectivity_ = {}
selectivity_['soma'] = sel_soma_all
selectivity_['dendrites'] = sel_dends_all
# Save the results
fname = f"{data_dir}/selectivity_best_models{seq_tag}.pkl"
with open(pathlib.Path(fname), 'wb') as handle:
    pickle.dump(selectivity_, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load and modify the post analysis data
fname_store = pathlib.Path(f"{dirname}/output_all_post_analysis_all_new")
with open(f'{fname_store}.pkl', 'rb') as file:
    results = pickle.load(file)

df_all = results['post_analysis']
df_all = fix_names(df_all)

df_modified = pd.DataFrame()
for model_name in model_to_keep:
    s = int(models_best[model_name][1])
    d = int(models_best[model_name][0])
    df_ = model_config(df_all, d, s, model_name)
    df_modified = pd.concat([df_modified, df_])

fname = f"{data_dir}/post_analysis_best_models{seq_tag}.pkl"
with open(pathlib.Path(fname), 'wb') as handle:
    pickle.dump(df_modified, handle, protocol=pickle.HIGHEST_PROTOCOL)

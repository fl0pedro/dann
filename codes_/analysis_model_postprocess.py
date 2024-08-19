#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:04:54 2023

@author: spiros
"""
import os
import keras
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from opt import get_data
from utils import information_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data_dir = "../DATA/"
# Simulation parameters
seq = False
tag = "_sequential" if seq else ""
datatype = "fmnist"  # datatype (mnist, fmnist, cifar10)
num_layers = 1
layers_list = range(2, (num_layers+1)*2+1, 2)

models = [
    'dend_ann_random', 'dend_ann_local_rfs',
    'dend_ann_global_rfs', 'dend_ann_all_to_all',
    'vanilla_ann', 'vanilla_ann_local_rfs',
    'vanilla_ann_global_rfs', 'vanilla_ann_random',
    'sparse_ann', 'sparse_ann_global_rfs',
    'sparse_ann_local_rfs', 'sparse_ann_all_to_all',
]

sigmas = [0.0]
trials = 5  # number of trials (i.e., initialization)
somata = [2**i for i in range(5, 10)]  # number of somata (i.e., somatic, second layer)
dendrites = [2**i for i in range(7)]  # number of dendrites per soma (i.e., dendritic layer)

save_data = True

s_all = {}
h_all = {}
df_post_analysis = None
columns_ = [
    'sparsity', 'entropy',
    'layer', 'model',
    'num_dends', 'num_soma',
    'trial'
]

missing_files = []
for model_type in models:
    s_all[f'{model_type}'] = {}
    h_all[f'{model_type}'] = {}
    for sigma in sigmas:
        for num_soma in somata:
            for num_dends in dendrites:
                s_all[f'{model_type}'][f'd_{num_dends}_s_{num_soma}'] = {}
                h_all[f'{model_type}'][f'd_{num_dends}_s_{num_soma}'] = {}
                for t in range(1, trials+1):

                    print(f"Analyze model: {model_type} with {num_soma} somata"
                          f" and {num_dends} dendrites for trials {t}, sigma {sigma}")
                    # Set up directory and file names
                    dirname = f"{data_dir}/results_{datatype}_{num_layers}_layer{tag}/"
                    print(f"\ndir: {dirname}")
                    subdir = f"{model_type}"
                    postfix = f"sigma_{sigma}_trial_{t}_dends_{num_dends}_soma_{num_soma}"
                    fname_model = Path(f"{dirname}/{subdir}/model_{postfix}.h5")
                    fname_data = Path(f"{dirname}/{subdir}/results_{postfix}.pkl")

                    if not os.path.exists(fname_data):
                        missing_files.append(fname_data)
                    # load the training/validation results
                    with open(fname_data, 'rb') as file:
                        DATA = pickle.load(file)

                    # Load your trained model
                    model = keras.models.load_model(fname_model)

                    # Get the data
                    data, labels, img_height, img_width, channels = get_data(
                        validation_split=0.1,
                        dtype=datatype,
                        normalize=True,
                        add_noise=True,
                        sigma=float(sigma),
                        seed=t
                    )
                    # Set the test-set data and labels to evaluate the model
                    x_test = data['test']
                    y_test = labels['test']

                    s_all[f'{model_type}'][f'd_{num_dends}_s_{num_soma}'][f'trial_{t}'] = []
                    h_all[f'{model_type}'][f'd_{num_dends}_s_{num_soma}'][f'trial_{t}'] = []

                    for layer in layers_list:
                        print(f"\n... layer {layer}\n")
                        features_list = [model.layers[layer].output]
                        feat_extraction_model = keras.Model(inputs=model.input, outputs=features_list)
                        # Get the activations for the sample input and translate them to numpy
                        activations_ = feat_extraction_model(x_test).numpy()
                        H, S, inactive, selectivity = information_metrics(activations_, y_test)
                        s_all[f'{model_type}'][f'd_{num_dends}_s_{num_soma}'][f'trial_{t}'].append(selectivity)
                        h_all[f'{model_type}'][f'd_{num_dends}_s_{num_soma}'][f'trial_{t}'].append(H)
                        # Create the DataFrame
                        if df_post_analysis is None:
                            df_post_analysis = pd.DataFrame(columns=columns_, index=[0])
                            df_post_analysis['sparsity'] = np.mean(S)
                            df_post_analysis['entropy'] = np.mean(H)
                            df_post_analysis['layer'] = 'dendritic' if layer == 2 else 'somatic'
                            df_post_analysis['trial'] = t
                            df_post_analysis['model'] = model_type
                            df_post_analysis['num_dends'] = num_dends
                            df_post_analysis['num_soma'] = num_soma
                        else:
                            df_ = pd.DataFrame(columns=columns_, index=[0])
                            df_['sparsity'] = np.mean(S)
                            df_['entropy'] = np.mean(H)
                            df_['layer'] = 'dendritic' if layer == 2 else 'somatic'
                            df_['trial'] = t
                            df_['model'] = model_type
                            df_['num_dends'] = num_dends
                            df_['num_soma'] = num_soma
                            df_post_analysis = pd.concat([df_post_analysis, df_], ignore_index=True)

if save_data:
    results = {}
    results['post_analysis'] = df_post_analysis
    results['selectivity'] = s_all
    results['entropy'] = h_all
    # Store data (serialize)
    fname_store =  Path(f"{dirname}/output_all_post_analysis_all_new")
    with open(f'{fname_store}.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

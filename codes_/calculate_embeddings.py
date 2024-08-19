#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:03:53 2024

@author: spiros
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import time
import keras
import pickle
import pathlib

from opt import get_data
from utils import _dim_reduction, neighborhood_hit

from plotting_functions import fix_names
from plotting_functions import keep_models
from plotting_functions import find_best_models
from plotting_functions import load_best_models
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness



t1 = time.time()

data_dir = "../DATA/"

datatype = "fmnist"
method = "tsne"
sigma = 0.0
num_layers = 1
num_rep = int(sys.argv[2])

seq = False
seq_tag = "_sequential" if seq else ""

# store figures
dirname_figs = '../FinalFigs_manuscript'
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

dirname = f"{data_dir}/results_{datatype}_{num_layers}_layer/"
fname_store =  pathlib.Path(f"{dirname}/output_all_final")
with open(f'{fname_store}.pkl', 'rb') as file:
    results = pickle.load(file)
# Keep models to plot, i.e., dend ANN and vanilla ANN.
model_to_keep = [
    'dANN-R', 'dANN-LRF', 'dANN-GRF', 'dANN-F',
    'sANN', 'sANN-LRF', 'sANN-GRF', 'sANN-F',
    'vANN-R', 'vANN-LRF', 'vANN-GRF',
    'vANN',
]

# Find the best models for each model_type for the default case scenario.
models_best = find_best_models(
    keep_models(fix_names(results['testing']), model_to_keep),
    model_to_keep,
    metric='accuracy',
    compare=True,
)

# Load data (deserialize)
dirname_models = f"{data_dir}/results_{datatype}_{num_layers}_layer{seq_tag}/"

# Find the best models for each model_type
model_names = [
    'dend_ann_random', 'dend_ann_local_rfs',
    'dend_ann_global_rfs', 'dend_ann_all_to_all',
    'sparse_ann', 'sparse_ann_global_rfs',
    'sparse_ann_local_rfs', 'sparse_ann_all_to_all',
    'vanilla_ann_random', 'vanilla_ann_local_rfs',
    'vanilla_ann_global_rfs', 'vanilla_ann',
]

all_models = load_best_models(
    models_best,
    model_names,
    dirname_models,
    sigma=sigma,
    trained=True if int(sys.argv[1]) == 1 else False
)

data, labels, img_height, img_width, channels = get_data(0.1, datatype)
x_train, y_train = data['train'], labels['train']
x_test, y_test = data['test'], labels['test']

n_data = y_test.shape[0] // 5
x_test, y_test = x_test[:n_data], y_test[:n_data]

embedding_model = {}
embedding_neurons = {}
scores_model = {}
n_trials = 5

for model_name in model_to_keep:
    print(f'\ncalculate for {model_name}')
    embedding_model[model_name] = {}
    embedding_neurons[model_name] = {}
    scores_model[model_name] = {}
    for t in range(n_trials):
        print(f'trial: {t}')
        embedding_model[model_name][f'trial_{t+1}'] = {}
        embedding_neurons[model_name][f'trial_{t+1}'] = {}
        scores_model[model_name][f'trial_{t+1}'] = {}
        for layer in [2, 4, 5]:
            print(f'layer: {layer}')
            model = all_models[model_name][t]
            features_list = [model.layers[layer].output]
            feat_extraction_model = keras.Model(inputs=model.input, outputs=features_list)
            # Get the activations for the sample input and translate them to numpy
            activations_ = feat_extraction_model(x_test).numpy()
            embedding = _dim_reduction(activations_, k=2, method=method)
            embedding_model[model_name][f'trial_{t+1}'][f'layer_{layer}'] = embedding
            scores_ = (silhouette_score(embedding, y_test),
                       neighborhood_hit(embedding, y_test, k=11),
                       trustworthiness(activations_, embedding, n_neighbors=11))
            scores_model[model_name][f'trial_{t+1}'][f'layer_{layer}'] = scores_
        print(f"Time needed for trial: {round(time.time() - t1, 2)} seconds\n")

results = {}
results['embeddings'] = embedding_model
results['scores'] = scores_model

str_ = "trained" if int(sys.argv[1]) == 1 else "untrained"
fname = f"{data_dir}/post_analysis_embeddings_{method}_{datatype}{seq_tag}_sigma_{sigma}_{str_}_rep_{num_rep}.pkl"
with open(fname, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nDone. Time needed for the analysis: {round((time.time() - t1)/60, 2)} minutes")

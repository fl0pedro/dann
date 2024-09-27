#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:22:14 2024

@author: spiros
"""

import os
import pickle
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from plotting_functions import fix_names, keep_models
from plotting_functions import my_style, calculate_best_model

# Set the seaborn style and color palette
sns.set_style("white")
plt.rcParams.update(my_style())

datatype = 'fmnist'
dirname_figs = '../FinalFigs_manuscript'
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

seq = False
seq_tag = "_sequential" if seq else ""

df_all_ = None
df_test_ = None

num_layers = 1

for lr in [0.01, 0.001, 0.0001]:
    if lr == 1e-3:
        data_dir = "../DATA/"
        dirname = f"{data_dir}/results_{datatype}_{num_layers}_layer{seq_tag}/"
    else:
        data_dir = "../DATA/explore_learning_rates/"
        dirname = f"{data_dir}/results_{datatype}_{num_layers}_layer{seq_tag}_lr_{lr}/"

    # Load data (deserialize)
    fname_store =  pathlib.Path(f"{dirname}/output_all_final")
    with open(f'{fname_store}.pkl', 'rb') as file:
        results = pickle.load(file)
    df_all = fix_names(results['training'])
    df_test = fix_names(results['testing'])

    df_all['lr'] = lr
    df_test['lr'] = lr

    if lr != 0.001:
        df_all['model'] += f"-{lr}"
        df_test['model'] += f"-{lr}"

    df_all_ = pd.concat([df_all_, df_all], ignore_index=True)
    df_test_ = pd.concat([df_test_, df_test], ignore_index=True)


palette = [
    '#8de5a1', '#ff9f9b', '#a1c9f4', '#d0bbff',
    '#d982b5', '#e0b0ff', '#8d8d8d'
]

models_to_keep = [
    'dANN-R',
    'dANN-LRF',
    'dANN-GRF',
    'pdANN',
    'vANN-0.01',
    'vANN-0.0001',
    'vANN',
]

df_all_ = keep_models(df_all_, models_to_keep)
df_test_ = keep_models(df_test_, models_to_keep)
df_all_['train_acc'] *= 100
df_all_['val_acc'] *= 100
df_test_['test_acc'] *= 100


# Create the figure
fig = plt.figure(
    num=4,
    figsize=(8.27*0.98, 11.69*0.25),
    layout='constrained'
    )

# Create layout
mosaic = [
    ["A", "B"],
]

axd = fig.subplot_mosaic(
    mosaic,
)

for label, ax in axd.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom')

# calculate the stats of
eval_metrics = calculate_best_model(df_test_[df_test_['model'] == 'vANN-0.0001'])

# Panel A
panel = "A"
sns.lineplot(
    data=df_test_,
    x="trainable_params",
    y="test_loss",
    hue="model",
    style="model",
    markers=True,
    dashes=False,
    ax=axd[panel],
    palette=palette,
)
axd[panel].axhline(
    np.min(eval_metrics, axis=0)[0],
    linestyle="--",
    c="k",
    linewidth=1.5,
)
axd[panel].axvline(
    eval_metrics[np.argmin(eval_metrics[:, 0])][-1],
    linestyle="--",
    c="k",
    linewidth=1.5,
)
axd[panel].set_ylabel("test loss")
axd[panel].set_xlabel("trainable params")
axd[panel].set_xscale("log")
axd[panel].set_yscale("log")

# Panel B
panel = "B"
sns.lineplot(
    data=df_test_,
    x="trainable_params",
    y="test_acc",
    hue="model",
    style="model",
    markers=True,
    dashes=False,
    legend=False,
    ax=axd[panel],
    palette=palette,
)
axd[panel].axhline(
    np.max(eval_metrics, axis=0)[1],
    linestyle="--",
    c="k",
    linewidth=1.5,
)
axd[panel].axvline(
    eval_metrics[np.argmax(eval_metrics[:, 1])][-1],
    linestyle="--",
    c="k",
    linewidth=1.5,
)
axd[panel].set_ylabel("test accuracy (%)")
axd[panel].set_xlabel("trainable params")
axd[panel].set_xscale("log")


# fig final format and save
figname = f"{dirname_figs}/supplementary_figure_4"
fig.savefig(
    pathlib.Path(f"{figname}.pdf"),
    bbox_inches='tight',
    dpi=600
)
fig.savefig(
    pathlib.Path(f"{figname}.svg"),
    bbox_inches='tight',
    dpi=600
)
fig.savefig(
    pathlib.Path(f"{figname}.png"),
    bbox_inches='tight',
    dpi=600
)
fig.show()

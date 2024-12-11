#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:37:17 2024

@author: spiros
"""

import os
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from plotting_functions import my_style, calc_eff_scores
from plotting_functions import keep_models

# Set the seaborn style and color palette
sns.set_style("white")
plt.rcParams.update(my_style())

palette = [
    '#8de5a1', '#ff9f9b', '#a1c9f4',
    '#d0bbff', '#8d8d8d'
]

dirname_figs = '../FinalFigs_manuscript'
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

data_dir = "../DATA/"

# Create the figure
fig = plt.figure(
    num=3,
    figsize=(7.086614*0.98, 11.69*0.5),
    layout='constrained'
)

# create the mosaic
axd = fig.subplot_mosaic(
    [["a", "b"],
     ["c", "d"],
     ],
    sharex=True
)

# add panel labels
for label, ax in axd.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(
        -20/72, 7/72,
        fig.dpi_scale_trans
    )
    ax.text(
        0.0, 1.0,
        label,
        transform=ax.transAxes + trans,
        fontsize='large', va='bottom'
    )

# load best models
fname_models = f"{data_dir}/all_datasets_best_models_final.pkl"
DATA = pd.read_pickle(open(fname_models, 'rb'))

# Keep models to plot, i.e., dend ANN and vanilla ANN.
models_to_keep = [
    'dANN-R',
    'dANN-LRF',
    'dANN-GRF',
    'pdANN',
    'vANN',
]
for key in DATA.keys():
    DATA[key] = keep_models(DATA[key], models_to_keep)

# Panel a
panel = "a"
sns.barplot(
    data=DATA["compare_acc"],
    x="data",
    y="trainable_params",
    hue="model",
    errorbar="sd",
    legend=False,
    palette=palette,
    ax=axd[panel]
)
axd[panel].set_yscale("log")
axd[panel].set_ylabel("trainable params")
axd[panel].set_title("matching vANN's highest accuracy")

# panel b
panel = "b"
sns.barplot(
    data=DATA["compare_loss"],
    x="data",
    y="trainable_params",
    hue="model",
    errorbar="sd",
    legend=False,
    palette=palette,
    ax=axd[panel]
)
axd[panel].set_yscale("log")
axd[panel].set_ylabel("trainable params")
axd[panel].set_title("matching vANN's minimum loss")

# normalized accuracy with loss
df_ = DATA["best_acc"].reset_index()
df_ = calc_eff_scores(df_, form='acc')

# panel c
panel = "c"
sns.barplot(
    data=df_,
    x="data",
    y="normed_acc",
    hue="model",
    palette=palette,
    errorbar=("sd", 1),
    legend=False,
    ax=axd[panel],
)
axd[panel].set_ylabel("accuracy eff score")
axd[panel].set_title("effeciency of the best model (highest accuracy)")

# normalized loss with params
df_ = DATA["best_acc"]
df_ = calc_eff_scores(df_, form='loss')

# panel d
panel = "d"
sns.barplot(
    data=df_,
    x="data",
    y="normed_loss",
    hue="model",
    errorbar=("sd", 1),
    legend=True,
    palette=palette,
    ax=axd[panel]
)
axd[panel].legend_.set_title(None)
axd[panel].set_ylabel("loss eff score")
axd[panel].set_title("effeciency of the best model (minimum loss)")


# fig final format and save
figname = f"{dirname_figs}/figure_3"
file_format = 'svg'
fig.savefig(
    pathlib.Path(f"{figname}.{file_format}"),
    bbox_inches='tight',
    dpi=600
)
fig.show()


# Calculate the losses, accuracies etc for Table 1.
df_ = DATA["best_loss"]
metric = "test_acc"
for m in ['dANN-R', 'dANN-LRF', 'dANN-GRF', 'pdANN', 'vANN']:
    print(m)
    for d in ['mnist']:
        print(np.round(np.mean(df_[(df_['model'] == m) & (df_['data'] == d)][metric]),3))
        print(np.round(np.std(df_[(df_['model'] == m) & (df_['data'] == d)][metric]), 4))

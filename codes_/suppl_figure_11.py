#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:18:13 2024

@author: spiros
"""

import os
import pickle
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from plotting_functions import fix_names, keep_models
from plotting_functions import my_style

# Set the seaborn style and color palette
# print(sns.color_palette("pastel3").as_hex())
sns.set_style("white")
plt.rcParams.update(my_style())

datatype = 'fmnist'
dirname_figs = '../FinalFigs_manuscript'
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

seq = False
seq_tag = "_sequential" if seq else ""
num_layers = 1
data_dir = "../DATA/"
dirname = f"{data_dir}/results_{datatype}_{num_layers}_layer{seq_tag}/"

# Create the figure
fig = plt.figure(
    num=11,
    figsize=(8.27*0.98, 11.69*0.35),
    layout='constrained'
    )

# Create layout
mosaic = [
    ["A", "B"],
]

axd = fig.subplot_mosaic(
    mosaic,
    sharex=True
)

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
        fontsize='large',
        va='bottom'
    )


# Load data (deserialize)
fname_store =  pathlib.Path(f"{dirname}/output_all_final")
with open(f'{fname_store}.pkl', 'rb') as file:
    results = pickle.load(file)
df_all = fix_names(results['training'])
df_test = fix_names(results['testing'])

# Keep models to plot, i.e., vanilla ANNs.

palette = [
    '#8de5a1', '#ff9f9b', '#a1c9f4', '#d0bbff',
]

models_to_keep = [
    'dANN-R',
    'dANN-LRF',
    'dANN-GRF',
    'dANN-F',
    'sANN',
    'sANN-LRF',
    'sANN-GRF',
    'sANN-F',
]

df_all_ = keep_models(df_all, models_to_keep)
df_test_ = keep_models(df_test, models_to_keep)
df_all_['train_acc'] *= 100
df_all_['val_acc'] *= 100
df_test_['test_acc'] *= 100


df_test_subtract = pd.DataFrame()

for m in range(4):
    m1 = models_to_keep[m]
    m2 = models_to_keep[m+4]

    df_1 = df_test_[df_test_["model"] == m1].reset_index()
    df_2 = df_test_[df_test_["model"] == m2].reset_index()
    df_test_subtract_ = pd.DataFrame()
    df_test_subtract_["test_loss"] = df_1["test_loss"] - df_2["test_loss"]
    df_test_subtract_["test_acc"] = df_1["test_acc"] - df_2["test_acc"]
    df_test_subtract_["model"] = f"Δ({m1}, {m2})"
    df_test_subtract_["trainable_params"] = df_1["trainable_params"]
    df_test_subtract = pd.concat([df_test_subtract, df_test_subtract_])

df_test_subtract = df_test_subtract.reset_index()



# Panel A
panel = "A"
sns.lineplot(
    data=df_test_subtract,
    x="trainable_params",
    y="test_loss",
    hue="model",
    style="model",
    markers=True,
    dashes=False,
    palette=palette,
    ax=axd[panel],
)

axd[panel].set_ylabel("Δ test loss")
axd[panel].set_xlabel("trainable params")
axd[panel].set_xscale("log")

# Panel B
panel = "B"
sns.lineplot(
    data=df_test_subtract,
    x="trainable_params",
    y="test_acc",
    hue="model",
    style="model",
    markers=True,
    dashes=False,
    legend=False,
    palette=palette,
    ax=axd[panel],
)
axd[panel].set_ylabel("Δ test accuracy (%)")
axd[panel].set_xlabel("trainable params")
axd[panel].set_xscale("log")


# fig final format and save
figname = f"{dirname_figs}/supplementary_figure_11"
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

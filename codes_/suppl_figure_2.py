#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:28:13 2024

@author: spiros
"""
import os
import pickle
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from plotting_functions import fix_names, keep_models
from plotting_functions import my_style


# Set the seaborn style and color palette
sns.set_style("white")
plt.rcParams.update(my_style())

# keep the same colors across figures
palette = [
    '#8de5a1', '#ff9f9b', '#a1c9f4',
    '#d0bbff', '#8d8d8d'
]

datatype = "fmnist"

dirname_figs = '../FinalFigs_manuscript/'
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

seq = False
seq_tag = "_sequential" if seq else ""
data_dir = "../DATA/"
dirname = f"{data_dir}/results_{datatype}_1_layer{seq_tag}/"

# Load data (deserialize)
fname_store =  pathlib.Path(f"{dirname}/output_all_final")  # to make it agnostic
with open(f'{fname_store}.pkl', 'rb') as file:
    results = pickle.load(file)
df_all = fix_names(results['training'])
df_test = fix_names(results['testing'])

# Keep models to plot, i.e., dend ANN and vanilla ANN.
models_to_keep = [
    'dANN-R',
    'dANN-LRF',
    'dANN-GRF',
    'dANN-F',
    'vANN',
]

df_all_ = keep_models(df_all, models_to_keep)
df_test_ = keep_models(df_test, models_to_keep)
df_all_['train_acc'] *= 100
df_all_['val_acc'] *= 100
df_test_['test_acc'] *= 100

df_all_ = df_all_[(df_all_['num_soma'] >= 32) & (df_all_['num_dends'] >= 1)]
y_ax = [
    "train_acc",
    "val_acc",
    "train_loss",
    "val_loss",
]
y_label = [
    "train accuracy (%)",
    "validation accuracy (%)",
    "train loss",
    "validation loss",
]
# Create the figure
fig = plt.figure(
    num=2,
    figsize=(8.27*0.98, 11.69*0.6),
    layout='constrained'
)

# create the mosaic
axd = fig.subplot_mosaic(
    [["A", "B"],
     ["C", "D"],
     ],
)

# add panel labels
for label, ax in axd.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes + trans,
        fontsize='large',
        va='bottom'
    )

for i, (model, (label, ax)) in enumerate(zip(models_to_keep, axd.items())):
    sns.lineplot(
        data=df_all_,
        x='epoch',
        y=y_ax[i],
        hue='model',
        style=df_all_[['num_soma', 'num_dends']].apply(tuple, axis=1),
        palette=palette,
        legend=False, # if i != 0 else True,
        ax=ax)

    ax.set_ylabel(y_label[i])

# fig final format and save
figname = f"{dirname_figs}/supplementary_figure_2.py"
fig.savefig(
    pathlib.Path(pathlib.Path(f"{figname}.pdf")),
    bbox_inches='tight',
    dpi=600
)
fig.savefig(
    pathlib.Path(pathlib.Path(f"{figname}.svg")),
    bbox_inches='tight',
    dpi=600
)
fig.savefig(
    pathlib.Path(pathlib.Path(f"{figname}.png")),
    bbox_inches='tight',
    dpi=600
)
fig.show()

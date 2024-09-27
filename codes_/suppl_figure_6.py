#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:10:24 2024

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

df_all_ = None
df_test_ = None
for num_layers in range(1, 5):
    if num_layers > 1:
        data_dir = "../DATA/explore_depth"
    else:
        data_dir = "../DATA/"

    if num_layers != 4:
        dirname = f"{data_dir}/results_{datatype}_{num_layers}_layer{seq_tag}/"
    else:
        dirname = f"{data_dir}/results_{datatype}_{num_layers}_layer{seq_tag}_lr_0.005/"

    # Load data (deserialize)
    fname_store =  pathlib.Path(f"{dirname}/output_all_final")
    with open(f'{fname_store}.pkl', 'rb') as file:
        results = pickle.load(file)
    df_all = fix_names(results['training'])
    df_test = fix_names(results['testing'])

    df_all['layers'] = num_layers
    df_test['layers'] = num_layers

    df_all_ = pd.concat([df_all_, df_all], ignore_index=True)
    df_test_ = pd.concat([df_test_, df_test], ignore_index=True)


palette = [
    '#8de5a1', '#ff9f9b', '#a1c9f4', '#d0bbff'
]

models_to_keep = [
    'dANN-R',
    'dANN-LRF',
    'dANN-GRF',
    'pdANN',
]

df_all_ = keep_models(df_all_, models_to_keep)
df_test_ = keep_models(df_test_, models_to_keep)
df_all_['train_acc'] *= 100
df_all_['val_acc'] *= 100
df_test_['test_acc'] *= 100


# Create the figure
fig = plt.figure(
    num=6,
    figsize=(8.27*0.98, 11.69*0.80),
    layout='constrained',
    )

# Create layout
mosaic = [
    ["A", "B"],
    ["C", "D"],
    ["E", "F"],
]

axd = fig.subplot_mosaic(
    mosaic,
    sharex=True,
)

for label, ax in axd.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(
        0.0, 1.0,
        label,
        transform=ax.transAxes + trans,
        fontsize='large',
        va='bottom'
    )

# Panel A
panel = "A"
sns.boxplot(
    data=df_test_[(df_test_['num_soma'] == 256) & (df_test_['num_dends'] == 8)],
    x="layers",
    y="test_loss",
    hue="model",
    ax=axd[panel],
    palette=palette,
    legend=False,
)
axd[panel].set_ylabel("test loss")
axd[panel].set_title("num dendrites per soma: 8")

# Panel B
panel = "B"
sns.boxplot(
    data=df_test_[(df_test_['num_soma'] == 256) & (df_test_['num_dends'] == 8)],
    x="layers",
    y="test_acc",
    hue="model",
    ax=axd[panel],
    palette=palette,
    legend=False,
)
axd[panel].set_ylabel("test accuracy (%)")


# Panel C
panel = "C"
sns.boxplot(
    data=df_test_[(df_test_['num_soma'] == 256) & (df_test_['num_dends'] == 16)],
    x="layers",
    y="test_loss",
    hue="model",
    ax=axd[panel],
    palette=palette,
    legend=False,
)
axd[panel].set_ylabel("test loss")
axd[panel].set_title("num dendrites per soma: 16")

# Panel D
panel = "D"
sns.boxplot(
    data=df_test_[(df_test_['num_soma'] == 256) & (df_test_['num_dends'] == 16)],
    x="layers",
    y="test_acc",
    hue="model",
    ax=axd[panel],
    palette=palette,
    legend=False,
)
axd[panel].set_ylabel("test accuracy (%)")

# Panel E
panel = "E"
sns.boxplot(
    data=df_test_[(df_test_['num_soma'] == 256) & (df_test_['num_dends'] == 32)],
    x="layers",
    y="test_loss",
    hue="model",
    ax=axd[panel],
    palette=palette,
    legend=False,
)
axd[panel].set_ylabel("test loss")
axd[panel].set_xlabel("dendrosomatic layers")
axd[panel].set_title("num dendrites per soma: 32")

# Panel F
panel = "F"
sns.boxplot(
    data=df_test_[(df_test_['num_soma'] == 256) & (df_test_['num_dends'] == 32)],
    x="layers",
    y="test_acc",
    hue="model",
    ax=axd[panel],
    palette=palette,
)
axd[panel].set_ylabel("test accuracy (%)")
axd[panel].set_xlabel("dendrosomatic layers")

# fig final format and save
figname = f"{dirname_figs}/supplementary_figure_6"
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

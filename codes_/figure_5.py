#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:25:37 2024

@author: spiros
"""

import os
import pickle
import pathlib
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from scipy.stats import skew, kurtosis
from plotting_functions import my_style, draw_text_metrics, short_to_long_names

parser = argparse.ArgumentParser()

parser.add_argument("source")
parser.add_argument("-o", "--output", default="../FinalFigs_manuscript")

args = parser.parse_args()

# Set the seaborn style and color palette
sns.set_style("white")
plt.rcParams.update(my_style())

palette = ['#8de5a1', '#ff9f9b', '#a1c9f4', '#b5b5ac']
palette2 = ['#409140', '#e06666', '#7abacc', '#8d8d8d']

datatype = 'fmnist'
dirname_figs = args.output
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

seq = False
seq_tag = "_sequential" if seq else ""
num_layers = 1
data_dir = args.source
dirname = f"{data_dir}/results_{datatype}_{num_layers}_layer{seq_tag}/"

# Create the figure
fig = plt.figure(
    num=5,
    figsize=(7.086614*0.98, 11.69*0.8),
    layout='tight'
)

# Split the figure in top and bottom
# Separate in three subfigures - one top, one middle, and one bottom
subfigs = fig.subfigures(
    nrows=3, ncols=1,
    height_ratios=[2, .8, .8],
    )

# Create the top left subfigure
mosaic = [
    ["a", "b", "c", "d"],
    ["e", "f", "g", "h"],
    ["i", "j", "k", "l"]
]

axt = subfigs[0].subplot_mosaic(
    mosaic,
    sharey=True,
    sharex=True,
    gridspec_kw={
        "hspace": 0.5,
    },
)

# label physical distance to the left and up:
trans = mtransforms.ScaledTranslation(
    -20/72, 7/72,
    fig.dpi_scale_trans
)

axt["a"].text(
    0.0, 1.0, "a",
    transform=axt["a"].transAxes + trans,
    fontsize='large',
    va='bottom'
)

# load weights of best models
fname_weights = f"{data_dir}/weights_best_models{seq_tag}.pkl"
with open(fname_weights, 'rb') as file:
    weights_loaded = pickle.load(file)

w_dend_all = weights_loaded['dendrites']
w_soma_all = weights_loaded['soma']
w_out_all = weights_loaded['output']

# Keep specific models
model_names = ['dANN-R', 'dANN-LRF', 'dANN-GRF', 'vANN']
models = short_to_long_names(model_names)

# Plot the results
xloc, yloc = 0.97, 0.97
for i, (model_type, (label, ax)) in enumerate(zip(models, list(axt.items())[:4])):
    sns.histplot(
        w_dend_all[f'{model_type}'].flatten(),
        bins=20,
        kde=True,
        stat='probability',
        ax=ax,
        color=palette[i],
    )
    ax.set_title(f"{model_names[i]}")
    ax.set_ylabel('probability')
    ax.set_xticks([])
    ax.set_xticklabels([])
    w_ = w_dend_all[f'{model_type}'].flatten()
    k = kurtosis(w_)
    s = skew(w_)
    r = max(w_) - min(w_)
    draw_text_metrics(ax, xloc, yloc, k, 'Kurtosis')
    draw_text_metrics(ax, xloc, yloc-0.2, s, 'Skewness')
    draw_text_metrics(ax, xloc, yloc-0.4, r, 'Range')


# Plot the results
for i, (model_type, (label, ax)) in enumerate(zip(models, list(axt.items())[4:8])):
    sns.histplot(
        w_soma_all[f'{model_type}'].flatten(),
        bins=20,
        kde=True,
        stat='probability',
        ax=ax,
        color=palette2[i],
    )
    ax.set_xlabel('weights')
    ax.set_ylabel('probability')
    ax.set_xticks([-2.5, 0.0, 2.5])
    ax.set_xticklabels([-2.5, 0.0, 2.5])
    w_ = w_soma_all[f'{model_type}'].flatten()
    k = kurtosis(w_)
    s = skew(w_)
    r = max(w_) - min(w_)
    draw_text_metrics(ax, xloc, yloc, k, 'Kurtosis')
    draw_text_metrics(ax, xloc, yloc-0.2, s, 'Skewness')
    draw_text_metrics(ax, xloc, yloc-0.4, r, 'Range')

# Plot the results
for i, (model_type, (label, ax)) in enumerate(zip(models, list(axt.items())[8:])):
    sns.histplot(
        w_out_all[f'{model_type}'].flatten(),
        bins=20,
        kde=True,
        stat='probability',
        ax=ax,
        color=palette2[i],
    )
    ax.set_xlabel('weights')
    ax.set_ylabel('probability')
    ax.set_xticks([-2.5, 0.0, 2.5])
    ax.set_xticklabels([-2.5, 0.0, 2.5])
    w_ = w_out_all[f'{model_type}'].flatten()
    k = kurtosis(w_)
    s = skew(w_)
    r = max(w_) - min(w_)
    draw_text_metrics(ax, xloc, yloc, k, 'Kurtosis')
    draw_text_metrics(ax, xloc, yloc-0.2, s, 'Skewness')
    draw_text_metrics(ax, xloc, yloc-0.4, r, 'Range')

# middle left subfigure
# load entropies of best models
fname_entropies = f"{data_dir}/entropies_best_models{seq_tag}.pkl"
with open(fname_entropies, 'rb') as file:
    entropies_loaded = pickle.load(file)

h_dend_all = entropies_loaded['dendrites']
h_soma_all = entropies_loaded['soma']

mosaic = [["a", "b", "c", "d"]]
axm = subfigs[1].subplot_mosaic(
    mosaic, sharey=True, sharex=True,
    gridspec_kw={
        "hspace": 0.5,
    },
)
trans = mtransforms.ScaledTranslation(
    -20/72, 7/72,
    fig.dpi_scale_trans
)

axm["a"].text(
    0.0, 1.0, "b",
    transform=axm["a"].transAxes + trans,
    fontsize='large',
    va='bottom'
)

for i, (model_type, (label, ax)) in enumerate(zip(models, axm.items())):
    sns.histplot(
        h_dend_all[f'{model_type}'],
        bins=20,
        kde=True,
        stat='probability',
        ax=ax,
        color=palette[i],
        label='dendritic',
        kde_kws=dict(bw_adjust=3),
    )
    sns.histplot(
        h_soma_all[f'{model_type}'],
        bins=20,
        kde=True,
        stat='probability',
        ax=ax,
        color=palette2[i],
        label='somatic',
        alpha=0.4,
        line_kws={'alpha': 0.6},
        kde_kws=dict(bw_adjust=3),
    )
    ax.set_xlabel('entropy (bits)')
    ax.set_ylabel('probability')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([0, 1, 2, 3])
    ax.legend() if i == 3 else None

# Make selectivity histograms - bottom left subfigure
fname_selectivity = f"{data_dir}/selectivity_best_models{seq_tag}.pkl"
with open(fname_selectivity, 'rb') as file:
    slectivity_loaded = pickle.load(file)

s_dend_all = slectivity_loaded['dendrites']
s_soma_all = slectivity_loaded['soma']

mosaic = [["a", "b", "c", "d"]]
axb = subfigs[2].subplot_mosaic(
    mosaic,
    sharey=True,
    sharex=True,
    gridspec_kw={
        "hspace": 0.5,
    },
)

trans = mtransforms.ScaledTranslation(
    -20/72, 7/72,
    fig.dpi_scale_trans
)

axb["a"].text(
    0.0, 1.0, "c",
    transform=axb["a"].transAxes + trans,
    fontsize='large',
    va='bottom'
)

n_classes = 10 if datatype != "emnist" else 47
for i, (model_type, (label, ax)) in enumerate(zip(models, axb.items())):
    sns.histplot(
        [x for x in s_dend_all[f'{model_type}'] if x > 0],
        bins=n_classes,
        kde=False,
        stat='probability',
        color=palette[i],
        label='dendritic',
        ax=ax,
    )
    sns.histplot(
        [x for x in s_soma_all[f'{model_type}'] if x > 0],
        bins=n_classes,
        kde=False,
        stat='probability',
        color=palette2[i],
        label='somatic',
        ax=ax,
    )
    ax.set_xlabel('classes')
    ax.set_ylabel('probability')
    xticks = [1, 5, 10] if datatype != 'emnist' else [1, 23, 47]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)

# fig final format and save
figname = f"{dirname_figs}/figure_5"
file_format = 'svg'
fig.savefig(
    pathlib.Path(f"{figname}.{file_format}"),
    bbox_inches='tight',
    dpi=600
)
fig.show()

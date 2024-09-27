#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:11:38 2024

@author: spiros
"""
import os
import pickle
import pathlib
import numpy as np
import seaborn as sns
import seaborn_image as isns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from plotting_functions import fix_names, keep_models
from plotting_functions import my_style, calculate_best_model

from opt import get_data

# Set the seaborn style and color palette
sns.set_style("white")
plt.rcParams.update(my_style())

palette = [
    '#8de5a1', '#ff9f9b', '#a1c9f4',
    '#d0bbff', '#8d8d8d'
]

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
    num=2,
    figsize=(8.27*0.98, 11.69*0.7),
    layout='constrained'
    )
# Separate in two subfigures - one top, one bottom
top_fig, bottom_fig = fig.subfigures(
    nrows=2, ncols=1,
    height_ratios=[1, 3.5]
    )
###############################################################################
# Top part of the figure
###############################################################################
mosaic_l = [["A", "B", "C", "D", "E"],
            ["F", "G", "H", "I", "J"],
            ]
axd_l = top_fig.subplot_mosaic(
    mosaic_l,
    gridspec_kw={
        "wspace": 0.0,
        "hspace": 0.0,
        "left": 0.0,
        },
    )

# label physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
axd_l["A"].text(0.0, 1.0, "A",
                transform=axd_l["A"].transAxes + trans,
                fontsize='large', va='bottom'
                )

data, label, img_height, img_width, channels = get_data(0.1, datatype)
x_train, y_train = data['train'], label['train']

for i, (labels, ax) in enumerate(axd_l.items()):
    x = x_train[y_train == i][0].reshape(
        img_width,
        img_height,
        channels
        ).squeeze()
    isns.imshow(
        x,
        gray=True if channels == 1 else False,
        cbar=False,
        square=True,
        ax=ax,
        )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.axis('off')

###############################################################################
# Bottom part of the figure
###############################################################################
# Create layout
mosaic = [["B", "C"],
          ["D", "E"],
          ]
axd = bottom_fig.subplot_mosaic(
    mosaic,
    )
for label, ax in axd.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom')


# Load data (deserialize)
fname_store =  pathlib.Path(f"{dirname}/output_all_final")
with open(f'{fname_store}.pkl', 'rb') as file:
    results = pickle.load(file)
df_all = fix_names(results['training'])
df_test = fix_names(results['testing'])

# Keep models to plot, i.e., dend ANN and vanilla ANN.
models_to_keep = [
    'dANN-R',
    'dANN-LRF',
    'dANN-GRF',
    'pdANN',
    'vANN',
]
df_all_ = keep_models(df_all, models_to_keep)
df_test_ = keep_models(df_test, models_to_keep)
df_all_['train_acc'] *= 100
df_all_['val_acc'] *= 100
df_test_['test_acc'] *= 100

# calculate the stats of
eval_metrics = calculate_best_model(df_test_[df_test_['model'] == 'vANN'])

# Panel B
panel = "B"
sns.lineplot(
    data=df_test_,
    x="trainable_params",
    y="test_loss",
    hue="model",
    style="model",
    markers=True,
    dashes=False,
    legend=False,
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
# axd[panel].set_yscale("log")
axd[panel].set_xlabel("trainable params")
axd[panel].set_xscale("log")

# Panel C
panel = "C"
sns.lineplot(
    data=df_test_,
    x="trainable_params",
    y="test_acc",
    hue="model",
    style="model",
    markers=True,
    dashes=False,
    legend=True,
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

# Data for the dendritic/somatic number
df = df_test_[(df_test_['num_soma'] > 200) & (df_test_['model'] != 'vANN')]

# Panel D
panel = "D"
sns.lineplot(
    data=df,
    y="test_loss",
    x="num_dends",
    hue="model",
    style="num_soma",
    markers=True,
    markersize=7,
    legend=False,
    ax=axd[panel],
    palette=palette[:-1],
)
axd[panel].axhline(
    np.nanmin(eval_metrics, axis=0)[0],
    linestyle="--",
    c="k",
    linewidth=1.5,
)
axd[panel].set_ylabel("test loss")
# axd[panel].set_yscale("log")
axd[panel].set_xlabel("dendrites per soma")
axd[panel].set_xscale("log", base=2)

# Panel E
panel = "E"
sns.lineplot(
    data=df,
    y="test_acc",
    x="num_dends",
    hue="model",
    style="num_soma",
    markers=True,
    markersize=7,
    ax=axd[panel],
    palette=palette[:-1],
)
axd[panel].axhline(
    np.nanmax(eval_metrics, axis=0)[1],
    linestyle="--",
    c="k",
    linewidth=1.5,
)
axd[panel].set_ylabel("test accuracy (%)")
axd[panel].set_xlabel("dendrites per soma")
axd[panel].set_xscale("log", base=2)

# fig final format and save
figname = f"{dirname_figs}/figure_2"
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:37:17 2024

@author: spiros
"""

import os
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from plotting_functions import my_style
from plotting_functions import keep_models

# Set the seaborn style and color palette
sns.set_style("white")
plt.rcParams.update(my_style())

data_dir = "../DATA"

palette = [
    '#8de5a1', '#ff9f9b', '#a1c9f4',
    '#d0bbff', '#8d8d8d'
]

dirname_figs = '../FinalFigs_manuscript'
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

# Create the figure
fig = plt.figure(
    num=7,
    figsize=(8.27*0.98, 11.69*0.5),
    layout='constrained'
)

# create the mosaic
axd = fig.subplot_mosaic(
    [["A"]],
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

# Panel A
panel = "A"
sns.boxplot(
    data=DATA["best_acc"],
    x="data",
    y="num_epochs_min",
    hue="model",
    palette=palette,
    ax=axd[panel]
)
axd[panel].set_ylabel("epochs")

axd[panel].axhline(15, linestyle="dashed", color='k')
axd[panel].axhline(25, linestyle="dashed", color='k')
axd[panel].axhline(50, linestyle="dashed", color='k')


# fig final format and save
figname = f"{dirname_figs}/supplementary_figure_7"
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

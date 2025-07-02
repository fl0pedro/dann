#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:42:32 2024

@author: spiros
"""
import os
import pickle
import pathlib
import argparse
import numpy as np
import seaborn as sns
import seaborn_image as isns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from opt import get_data, perturb_array

from plotting_functions import my_style
from plotting_functions import fix_names
from plotting_functions import keep_models
from plotting_functions import calc_eff_scores
from plotting_functions import find_best_models
from plotting_functions import keep_best_models_data

parser = argparse.ArgumentParser()

parser.add_argument("source")
parser.add_argument("-o", "--output", default="../FinalFigs_manuscript")

args = parser.parse_args()

# Set the seaborn style and color palette
# print(sns.color_palette("pastel3").as_hex())
sns.set_style("whitegrid")
plt.rcParams.update(my_style())

palette = [
    '#8de5a1', '#ff9f9b', '#a1c9f4',
    '#8d8d8d'
]

dirname_figs = args.output
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

# Create the figure
fig = plt.figure(
    num=7,
    figsize=(7.086614*0.98, 11.69*0.6),
    layout='constrained'
    )
# Split the figure in top and bottom
# Separate in three subfigures - one top, one middle, and one bottom
subfigs = fig.subfigures(
    nrows=2, ncols=2,
    width_ratios=[1, 3]
    )

datatype = 'fmnist'

mosaic = [["a", "b"],
          ["c", "d"],
          ]

axt = subfigs[0, 0].subplot_mosaic(
    mosaic,
    gridspec_kw={
    "hspace": 0.1,
    "wspace":0.1
    },
)

# label physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
axt["a"].text(
    0.0, 1.0, "a",
    transform=axt["a"].transAxes + trans,
    fontsize='large', va='bottom'
)

data, labels, img_height, img_width, channels = get_data(
    validation_split=0.1,
    dtype=datatype,
)
x_train = data['train']
y_train = labels['train']
x = x_train[0].reshape(img_width, img_height, channels).squeeze()
sigmas = [0.25, 0.5, 0.75, 1.0]

for s, (labels, ax) in zip(sigmas, axt.items()):
    pertrubation = np.random.normal(loc=0.0, scale=s, size=x.shape)
    x_noise = perturb_array(x, pertrubation)
    isns.imshow(
        x_noise,
        gray=True if channels == 1 else False,
        cbar=False,
        square=True,
        ax=ax,
        )
    ax.set_title(f'σ={s}')
    # Remove x-axis and y-axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.grid(False)


num_layers = 1
data_dir = args.source
dirname = f"{data_dir}/results_{datatype}_{num_layers}_layer/"

fname_store =  pathlib.Path(f"{dirname}/output_all_final")
with open(f'{fname_store}.pkl', 'rb') as file:
    results = pickle.load(file)
# Keep models to plot, i.e., dend ANN and vanilla ANN.
models_to_keep = [
    'dANN-R',
    'dANN-LRF',
    'dANN-GRF',
    'vANN',
]
# Find the best models for each model_type for the default case scenario.
models_best = find_best_models(
    keep_models(fix_names(results['testing']), models_to_keep),
    models_to_keep,
    metric='accuracy',
    compare=True,
)

# Take the noise data and load the results for the best models.
fname_store =  pathlib.Path(f"{dirname}/output_all_noise_final")
with open(f'{fname_store}.pkl', 'rb') as file:
    results_noise = pickle.load(file)

df_test_u = keep_best_models_data(
    keep_models(fix_names(results_noise['testing']), models_to_keep),
    models_best
)

# normalize the accuracy and loss metrics.
df_test_u["data"] = "fmnist"
df_test_u['test_acc'] *= 100
df_test_u = calc_eff_scores(df_test_u, form='acc')
df_test_u = calc_eff_scores(df_test_u, form='loss')

mosaic = [["a", "b"],
          ]

axt = subfigs[0, 1].subplot_mosaic(
    mosaic,
    gridspec_kw={
    "hspace": 0.1,
    "wspace":0.1
    },
    )

# label physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
axt["a"].text(
    0.0, 1.0, "b",
    transform=axt["a"].transAxes + trans,
    fontsize='large', va='bottom'
)

panel = "a"
sns.lineplot(
    data=df_test_u,
    x="sigma",
    y="normed_loss",
    hue="model",
    errorbar=("sd", 1),
    palette=palette,
    legend=False,
    ax=axt[panel])
axt[panel].set_xlabel("noise level (σ)")
axt[panel].set_ylabel("loss eff score")
axt[panel].grid(False)

panel = "b"
sns.lineplot(
    data=df_test_u,
    x="sigma",
    y="normed_acc",
    hue="model",
    errorbar=("sd", 1),
    palette=palette,
    legend=True,
    ax=axt[panel])
axt[panel].legend_.set_title(None)
axt[panel].set_xlabel("noise level (σ)")
axt[panel].set_ylabel("accuracy eff score")
axt[panel].grid(False)


mosaic = [["a", ".", "b"],
          ]

axt = subfigs[1, 0].subplot_mosaic(
    mosaic,
    gridspec_kw={
    "hspace": 0.2,
    "wspace":0.2
    },
    )
# label physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
axt["a"].text(
    0.0, 1.0, "c",
    transform=axt["a"].transAxes + trans,
    fontsize='large', va='bottom'
)


x1 = x_train[y_train == 0][0].reshape(img_width, img_height, channels).squeeze()
x2 = x_train[y_train == 9][0].reshape(img_width, img_height, channels).squeeze()
for x, (label, ax) in zip([x1, x2], axt.items()):
    isns.imshow(
        x,
        gray=True if channels == 1 else False,
        cbar=False,
        square=True,
        ax=ax,
        )
    # Remove x-axis and y-axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.grid(False)


num_layers = 1
dirname = f"{data_dir}/results_{datatype}_{num_layers}_layer_sequential/"

fname_store =  pathlib.Path(f"{dirname}/output_all_final")
with open(f'{fname_store}.pkl', 'rb') as file:
    results_seq = pickle.load(file)

df_test_s = keep_best_models_data(
    keep_models(fix_names(results_seq['testing']), models_to_keep),
    models_best
)

df_test_s["data"] = "fmnist"
df_test_s['test_acc'] *= 100
df_test_s = calc_eff_scores(df_test_s, form='acc')
df_test_s = calc_eff_scores(df_test_s, form='loss')


mosaic = [["a", "b"],
          ]

axt = subfigs[1, 1].subplot_mosaic(
    mosaic,
    gridspec_kw={
    "hspace": 0.1,
    "wspace":0.1
    },
    )

# label physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
axt["a"].text(0.0, 1.0, "d",
                transform=axt["a"].transAxes + trans,
                fontsize='large', va='bottom'
                )

panel = "a"
sns.barplot(
    data=df_test_s,
    x="model",
    y="normed_loss",
    hue="model",
    errorbar=("sd", 1),
    palette=palette,
    ax=axt[panel])
axt[panel].set_xticks([])
axt[panel].set_ylabel("loss eff score")
axt[panel].set_yscale("log")
axt[panel].grid(False)

panel = "b"
sns.barplot(
    data=df_test_s,
    x="model",
    y="normed_acc",
    hue="model",
    errorbar=("sd", 1),
    palette=palette,
    ax=axt[panel])
axt[panel].set_xticks([])
axt[panel].set_ylabel("accuracy eff score")
axt[panel].grid(False)


# fig final format and save
figname = f"{dirname_figs}/figure_7"
file_format = 'svg'
fig.savefig(
    pathlib.Path(f"{figname}.{file_format}"),
    bbox_inches='tight',
    dpi=600
)
fig.show()

# Print results for Supplementary Table 2
df_ = df_test_u.copy()
for m in ['dANN-R', 'dANN-LRF', 'dANN-GRF', 'vANN']:
    print(f"\n{m}")
    for d in [0.0, 0.25, 0.5, 0.75, 1.0]:
        df__ = df_[(df_['model'] == m) & (df_['sigma'] == d)].copy()
        print(np.round(np.mean(df__['test_loss']), 4))
        print(np.round(np.std(df__['test_loss']), 4))


# Print results for Table 2
df_ = df_test_s.copy()
for m in ['dANN-R', 'dANN-LRF', 'dANN-GRF', 'vANN']:
    print(f"\n{m}")
    for d in [0.0]:
        print(np.round(np.mean(df_[df_['model'] == m]['test_acc']), 3))
        print(np.round(np.std(df_[df_['model'] == m]['test_acc']), 4))

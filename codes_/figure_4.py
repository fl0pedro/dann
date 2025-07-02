#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:44:19 2024

@author: spiros
"""

import os
import pathlib
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from plotting_functions import my_style, calc_eff_scores
from plotting_functions import keep_models

parser = argparse.ArgumentParser()

parser.add_argument("source")
parser.add_argument("-o", "--output", default="../FinalFigs_manuscript")

args = parser.parse_args()

# Set the seaborn style and color palette
sns.set_style("white")
plt.rcParams.update(my_style())

palette = [
    '#8de5a1', '#467250',
    '#ff9f9b', '#7f4f4d',
    '#a1c9f4', '#607892',
    '#d0bbff', '#7c7099'
]

dirname_figs = args.output
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

num_layers = 1
data_dir = args.source

# Create the figure
fig = plt.figure(
    num=4,
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
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(
        0.0, 1.0,
        label,
        transform=ax.transAxes + trans,
        fontsize='large',
        va='bottom'
    )

# load best models
fname_models = f"{data_dir}/all_datasets_best_models_final.pkl"
DATA = pd.read_pickle(open(fname_models, 'rb'))

models_to_keep = [
    'dANN-R',
    'vANN-R',
    'dANN-LRF',
    'vANN-LRF',
    'dANN-GRF',
    'vANN-GRF',
    'pdANN',
    'vANN',
]

for key in DATA.keys():
    DATA[key] = keep_models(DATA[key], models_to_keep)


df_test_subtract = pd.DataFrame()
df_test = DATA["compare_acc"]
for data in df_test["data"].unique():
    df_test_ = df_test[df_test["data"] == data]
    df_test_subtract_ = pd.DataFrame()
    for m in range(4):
        m1 = models_to_keep[m]
        m2 = models_to_keep[m+4]

        df_1 = df_test_[df_test_["model"] == m1].reset_index()
        df_2 = df_test_[df_test_["model"] == m2].reset_index()
        df_test_subtract_["trainable_params"] = df_1["trainable_params"] - df_2["trainable_params"]
        df_test_subtract_["model"] = f"Δ({m1}, {m2})"
        df_test_subtract_["data"] = data
        df_test_subtract = pd.concat([df_test_subtract, df_test_subtract_])

df_test_subtract = df_test_subtract.reset_index()


# panel a
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

df_test_subtract = pd.DataFrame()
df_test = DATA["compare_loss"]
for data in df_test["data"].unique():
    df_test_ = df_test[df_test["data"] == data]
    df_test_subtract_ = pd.DataFrame()
    for m in range(4):
        m1 = models_to_keep[m]
        m2 = models_to_keep[m+4]

        df_1 = df_test_[df_test_["model"] == m1].reset_index()
        df_2 = df_test_[df_test_["model"] == m2].reset_index()
        df_test_subtract_["trainable_params"] = df_1["trainable_params"] - df_2["trainable_params"]
        df_test_subtract_["model"] = f"Δ({m1}, {m2})"
        df_test_subtract_["data"] = data
        df_test_subtract = pd.concat([df_test_subtract, df_test_subtract_])

df_test_subtract = df_test_subtract.reset_index()

# panel b
panel = "b"
sns.barplot(
    data=DATA["compare_loss"],
    x="data",
    y="trainable_params",
    hue="model",
    errorbar="sd",
    legend=True,
    palette=palette,
    ax=axd[panel])
axd[panel].set_yscale("log")
axd[panel].set_ylabel("trainable params")
axd[panel].set_title("matching vANN's minimum loss")


palette = [
    '#8de5a1',
    '#ff9f9b',
    '#a1c9f4',
    '#d0bbff',
]

models_to_keep = [
    'dANN-R',
    'dANN-LRF',
    'dANN-GRF',
    'pdANN',
    'vANN-R',
    'vANN-LRF',
    'vANN-GRF',
    'vANN',
]

# normalized accuracy
df_test_subtract = pd.DataFrame()
df_test = DATA["best_acc"]
df_test = calc_eff_scores(df_test, form='acc')
for data in df_test["data"].unique():
    df_test_ = df_test[df_test["data"] == data]
    df_test_subtract_ = pd.DataFrame()
    for m in range(4):
        m1 = models_to_keep[m]
        m2 = models_to_keep[m+4]

        df_1 = df_test_[df_test_["model"] == m1].reset_index()
        df_2 = df_test_[df_test_["model"] == m2].reset_index()
        df_test_subtract_["normed_acc"] = df_1["normed_acc"] - df_2["normed_acc"]
        df_test_subtract_["model"] = f"Δ({m1}, {m2})"
        df_test_subtract_["data"] = data
        df_test_subtract = pd.concat([df_test_subtract, df_test_subtract_])

df_test_subtract = df_test_subtract.reset_index()

# panel c
panel = "c"
sns.barplot(
    data=df_test_subtract,
    x="data",
    y="normed_acc",
    hue="model",
    palette=palette,
    errorbar=("sd"),
    legend=False,
    ax=axd[panel],
    )
axd[panel].set_ylabel("Δ accuracy eff score")
axd[panel].set_title("best models")

# normalized accuracy
df_test_subtract = pd.DataFrame()
df_test = DATA["best_acc"]
df_test = calc_eff_scores(df_test, form='loss')
for data in df_test["data"].unique():
    df_test_ = df_test[df_test["data"] == data]
    df_test_subtract_ = pd.DataFrame()
    for m in range(4):
        m1 = models_to_keep[m]
        m2 = models_to_keep[m+4]

        df_1 = df_test_[df_test_["model"] == m1].reset_index()
        df_2 = df_test_[df_test_["model"] == m2].reset_index()
        df_test_subtract_["normed_loss"] = df_1["normed_loss"] - df_2["normed_loss"]
        df_test_subtract_["model"] = f"Δ({m1}, {m2})"
        df_test_subtract_["data"] = data
        df_test_subtract = pd.concat([df_test_subtract, df_test_subtract_])

df_test_subtract = df_test_subtract.reset_index()

# panel d
panel = "d"
sns.barplot(
    data=df_test_subtract,
    x="data",
    y="normed_loss",
    hue="model",
    errorbar=("sd"),
    legend=True,
    palette=palette,
    ax=axd[panel])
axd[panel].legend_.set_title(None)
axd[panel].set_ylabel("Δ loss eff score")
axd[panel].set_title("best models")


# fig final format and save
figname = f"{dirname_figs}/figure_4"
file_format = 'svg'
fig.savefig(
    pathlib.Path(f"{figname}.{file_format}"),
    bbox_inches='tight',
    dpi=600
)
fig.show()

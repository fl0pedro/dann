#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:16:41 2024

@author: spiros
"""
import os
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from opt import get_data
from plotting_functions import my_style
from plotting_functions import get_class_names
from plotting_functions import calculate_proj_scores


# Set the seaborn style and color palette
sns.set_style("white")
plt.rcParams.update(my_style())

palette = ['#8de5a1', '#ff9f9b', '#a1c9f4', '#b5b5ac']
palette2 = ['#409140', '#e06666', '#7abacc', '#8d8d8d']

dirname_figs = '../FinalFigs_manuscript'
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

datatype = 'fmnist'
dim_method = 'tsne'
seq = False
seq_tag = "_sequential" if seq else ""

# Keep models to plot, i.e., dend ANN and vanilla ANN.
model_to_keep = [
    'dANN-R',
    'dANN-LRF',
    'dANN-GRF',
    'vANN'
]

data, labels, img_height, img_width, channels = get_data(0.1, datatype)
n_data = labels['test'].shape[0] // 5
x_test = data['test'][:n_data]
y_test = labels['test'][:n_data]
n_classes = len(set(y_test))
y_test_class_names = get_class_names(datatype, y_test)


sigma = 0.0
df_scores_trained_all = None

reps = 10
for nrep in range(1, reps+1):
    df_scores_trained, embeddings_dict = calculate_proj_scores(
        model_to_keep, "../DATA/", sigma=0.0,
        dim_method=dim_method, datatype=datatype,
        seq_tag=seq_tag, learn_phase="trained", rep=nrep)
    df_scores_trained['repetition'] = nrep

    df_scores_trained_all = pd.concat([df_scores_trained_all, df_scores_trained])

# Create the figure
fig = plt.figure(
    num=6,
    figsize=(8.27*0.98, 11.69*0.7),
    layout='constrained'
    )

subfigs = fig.subfigures(
    nrows=1, ncols=2,
    width_ratios=[3, 1]
    )

# create the mosaic
axd = subfigs[0].subplot_mosaic(
    [["A", "B",],
     ["C", "D",],
     ["E", "F",],
     ["G", "H",],],
    sharex=True,
    sharey=True,
    gridspec_kw={
    "hspace": 0.2,
    "wspace":0.2
    },
)

# add panel labels
mos = ["A", "B", "C", "D"]
mos_ = ["A", "C", "E", "G"]
axd_ = {k: v for k, v in axd.items() if k in mos_}
for i, (label, ax) in enumerate(axd_.items()):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, mos[i], transform=ax.transAxes + trans,
            fontsize='large', va='bottom')


for i, (layer, ax) in enumerate(axd.items()):
    k = 2 if i % 2 == 0 else 4
    embeddings = embeddings_dict[model_to_keep[i // 2]]['trial_1'][f'layer_{k}']
    sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        s=7,
        hue=y_test_class_names,
        palette="tab10",
        legend=False,
        ax=ax)
    ax.set_xlabel(f"{dim_method.upper()} dim 1")
    ax.set_ylabel(f"{dim_method.upper()} dim 2")
    ax.set_title(model_to_keep[i // 2])


df_scores_trained = df_scores_trained_all[df_scores_trained_all['layer'] != 'output']
df_scores_ = df_scores_trained.copy()
df_scores_['model+layer'] = df_scores_['model'] + df_scores_['layer']

# make a combined palette
p = np.concatenate([np.array(palette).reshape(1,-1),
                    np.array(palette2).reshape(1,-1)],
                   axis=0)
palette3 = list(p.T.flatten())

# create the mosaic
axd = subfigs[1].subplot_mosaic(
    [["E"], ["F"], ["G"]],
    gridspec_kw={
    "hspace": 0.2,
    "wspace":0.2
    },
)

# add panel labels
mos = ["E", "F", "G"]
for i, (label, ax) in enumerate(axd.items()):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, mos[i], transform=ax.transAxes + trans,
            fontsize='large', va='bottom')

panel = "E"
sns.barplot(
    data=df_scores_,
    x=None,
    y="silhouette",
    hue="model+layer",
    legend=False,
    palette=palette3,
    ax=axd[panel]
)
axd[panel].set_xticks([])
axd[panel].set_xticklabels([])
axd[panel].set_ylabel('Silhouette score')
axd[panel].set_ylim([0.0, 0.35])

panel = "F"
sns.barplot(
    data=df_scores_,
    x=None,
    y="nh_score",
    hue="model+layer",
    legend=False,
    palette=palette3,
    ax=axd[panel]
)
axd[panel].set_xticks([])
axd[panel].set_xticklabels([])
axd[panel].set_ylabel('NH score')
axd[panel].set_ylim([0.6, 0.85])

panel = "G"
sns.barplot(
    data=df_scores_,
    x=None,
    y="trustworthiness",
    hue="model+layer",
    legend=False,
    palette=palette3,
    ax=axd[panel]
)
axd[panel].set_xticks([])
axd[panel].set_xticklabels([])
axd[panel].set_ylabel('trustworthiness')
axd[panel].set_ylim([0.96, 1.0])

# fig final format and save
figname = f"{dirname_figs}/figure_6"
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




# Statistical analysis
import pingouin as pg

aov_sil = pg.anova(
    data=df_scores_,
    dv='silhouette',
    between=['model', 'layer'],
    detailed=True
)

pair_sil = pg.pairwise_tests(
    data=df_scores_,
    dv='silhouette',
    between=['model','layer'],
    padjust='bonf'
)

print("\nStats in Silhouette")
print(aov_sil)
print(pair_sil)

aov_nh = pg.anova(
    data=df_scores_,
    dv='nh_score',
    between=['model', 'layer'],
    detailed=True
)
pair_nh = pg.pairwise_tests(
    data=df_scores_,
    dv='nh_score',
    between=['model','layer'],
    padjust='bonf'
)

print("\nStats in nh score")
print(aov_nh)
print(pair_nh)

aov_trust = pg.anova(
    data=df_scores_,
    dv='trustworthiness',
    between=['model', 'layer'],
    detailed=True
)

pair_trust = pg.pairwise_tests(
    data=df_scores_,
    dv='trustworthiness',
    between=['model','layer'],
    padjust='bonf'
)

print("\nStats in trustworthiness")
print(aov_trust)
print(pair_trust)

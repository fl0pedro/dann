#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 23:34:25 2024

@author: spiros
"""
import os
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt

from plotting_functions import my_style, make_subplots

# Set the seaborn style and color palette
sns.set_style("white")
# sns.set_context("paper")
plt.rcParams.update(my_style())

dirname_figs = '../FinalFigs_manuscript'
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

# Create the figure
fig = plt.figure(
    num=1,
    figsize=(8.27*0.98, 11.69*0.9),
    layout='constrained'
)

# Separate in two subfigures - one top, one bottom
top_fig, middle_fig, bottom_fig = fig.subfigures(
    nrows=3, ncols=1,
    height_ratios=[1, 1, 2]
)

###############################################################################
# Top part of the figure
###############################################################################
left_fig, right_fig = top_fig.subfigures(
    nrows=1, ncols=2,
)

make_subplots(fig, fig_part=left_fig, dataset="mnist", label="A")
make_subplots(fig, fig_part=right_fig, dataset="fmnist", label="B")

###############################################################################
# Middle part of the figure
###############################################################################
left_fig, right_fig = middle_fig.subfigures(
    nrows=1, ncols=2,
    )

make_subplots(fig, fig_part=left_fig, dataset="kmnist", label="C")
make_subplots(fig, fig_part=right_fig, dataset="cifar10", label="D")

###############################################################################
# Bottom part of the figure
###############################################################################
make_subplots(fig, fig_part=bottom_fig, dataset="emnist", label="E")

# fig final format and save
figname = f"{dirname_figs}/supplementray_figure_1"
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:12:14 2021

@author: spiros
"""
import pickle
import pathlib
import keras
import numpy as np
import pandas as pd
import seaborn_image as isns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms

from opt import get_data


def symmetrical_colormap(cmap_settings, new_name=None):
    """
    Make symmetrical colormaps.

    This function takes a colormap and create a new one,
    as the concatenation of itself by a symmetrical fold.
    https://stackoverflow.com/questions/28439251/symmetric-colormap-matplotlib

    Parameters
    ----------
    cmap_settings : TUPLE
        Tuple with two elements. First, the colormap, and second the
        discretization factor. Example: cmap_settings = ('Blues', None)
        provide int instead of None to "discretize/bin" the colormap
    new_name : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    mymap : TYPE
        DESCRIPTION.

    """
    # get the colormap
    cmap = plt.cm.get_cmap(*cmap_settings)
    if not new_name:
        new_name = "sym_"+cmap_settings[0]  # ex: 'sym_Blues'
    # this defined the roughness of the colormap, 128 fine
    n = 128
    # get the list of color from colormap
    # take the standard colormap # 'right-part'
    colors_r = cmap(np.linspace(0, 1, n))
    # take the first list of color and flip the order # "left-part"
    colors_l = colors_r[::-1]

    # combine them and build a new colormap
    colors = np.vstack((colors_l, colors_r))
    mymap = mcolors.LinearSegmentedColormap.from_list(new_name, colors)

    return mymap


def my_style():
    """
    Create custom plotting style.

    Returns
    -------
    my_style : dict
        Dictionary with matplotlib parameters.

    """
    # color pallette
    fsize = 10
    my_style = {
        # Use LaTeX to write all text
        "text.usetex": False,
        "font.family": "Arial",
        # "font.weight": "bold",
        # Use 16pt font in plots, to match 16pt font in document
        "axes.labelsize": fsize,
        "axes.titlesize": fsize,
        "font.size": fsize,
        "grid.color": "black",
        "grid.linewidth": 0.2,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": fsize-2,
        "xtick.labelsize": fsize,
        "ytick.labelsize": fsize,
        "axes.linewidth": 1.5,
        "lines.markersize": 4.0,
        "lines.linewidth": 1.0,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "axes.edgecolor": "black",
        # "axes.labelweight": "bold",
        # "axes.titleweight": "bold",   # Add this line to set the title font weight to bold
        "axes.spines.right": False,
        "axes.spines.top": False,
        "svg.fonttype": "none"
    }

    return my_style


def set_size(width, fraction=1):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
        Document textwidth or columnwidth in pts
    fraction: float, optional
        Fraction of the width which you wish the figure to occupy
    Returns
    -------
    fig_dim: tuple
        Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


def keep_models(df_all, names):
    df = pd.DataFrame()
    for n in names:
        df = pd.concat([df,  df_all[(df_all['model'] == n)]])
    return df


def fix_names(df):
    df = df.replace(['dend_ann_global_rfs'], 'dANN-GRF')
    df = df.replace(['dend_ann_local_rfs'], 'dANN-LRF')
    df = df.replace(['dend_ann_random'], 'dANN-R')
    df = df.replace(['dend_ann_all_to_all'], 'pdANN')
    df = df.replace(['vanilla_ann'], 'vANN')
    df = df.replace(['vanilla_ann_random'], 'vANN-R')
    df = df.replace(['vanilla_ann_local_rfs'], 'vANN-LRF')
    df = df.replace(['vanilla_ann_global_rfs'], 'vANN-GRF')
    df = df.replace(['sparse_ann'], 'sANN')
    df = df.replace(['sparse_ann_local_rfs'], 'sANN-LRF')
    df = df.replace(['sparse_ann_global_rfs'], 'sANN-GRF')
    df = df.replace(['sparse_ann_all_to_all'], 'psANN')
    df = df.replace(['vanilla_ann_dropout_0.2'], 'vANN-0.2')
    df = df.replace(['vanilla_ann_dropout_0.5'], 'vANN-0.5')
    df = df.replace(['vanilla_ann_dropout_0.8'], 'vANN-0.8')
    return df


def calculate_best_model(df):
    """
    Calculate the stats for the vanilla dnn.

    Parameters
    ----------
    df : pandas.DataFrame
        DESCRIPTION.

    Returns
    -------
    losses : TYPE
        DESCRIPTION.
    accs : TYPE
        DESCRIPTION.

    """
    dendrites = [2**i for i in range(7)]
    somata = [2**i for i in range(5, 10)]
    eval_metrics = []
    for d in dendrites:
        for s in somata:
            df_ = df[(df["num_dends"] == d) & (df["num_soma"] == s)]
            eval_metrics.append(
                [np.mean(df_['test_loss']),
                 np.mean(df_['test_acc']), d, s,
                 np.mean(df_['trainable_params'])],
                )
    return np.array(eval_metrics)


def model_config(df, d, s, m):
    return(df[(df['num_dends'] == d) & (df['num_soma'] == s) & (df['model'] == m)])


def keep_best_models_data(df, models_list):
    df_ = pd.DataFrame()
    for model in models_list.keys():
        d, s = int(models_list[model][0]), int(models_list[model][1])
        df_ = pd.concat([df_, model_config(df, d, s, model)])
    return df_.reset_index()


def load_models(model_type, num_dends, num_soma, dirname,
                sigma, trained, n_trials=5,):
    models_list = []
    for t in range(1, n_trials+1):
        postfix = f"sigma_{sigma}_trial_{t}_dends_{num_dends}_soma_{num_soma}"
        if trained:
            fname_model = pathlib.Path(f"{dirname}/{model_type}/model_{postfix}.h5")
        else:
            fname_model = pathlib.Path(f"{dirname}/{model_type}/untrained_model_{postfix}.h5")
        models_list.append(keras.models.load_model(fname_model))
    return models_list


def load_best_models(model_list, names, dirname, sigma=0.0, trained=True):
    models_all = {}
    for model, name in zip(model_list.keys(), names):
        d, s = int(model_list[model][0]), int(model_list[model][1])
        models_all[model] = load_models(
            name, d, s,
            dirname,
            sigma=sigma,
            trained=trained,
            )
    return models_all


def find_best_models(df, model_names, metric='accuracy', compare=True,
                     baseline='vANN'):

    model_names_ = [n for n in model_names if n != baseline]
    if compare:
        eval_metrics_base = calculate_best_model(df[df['model'] == baseline])
        models_best = {}
        if metric == 'accuracy':
            best_acc = eval_metrics_base[np.argmax(eval_metrics_base[:, 1])][1]
            for model_type in model_names_:
                eval_metrics = calculate_best_model(df[df['model'] == model_type])
                for metric_ in eval_metrics:
                    if metric_[1] > best_acc:
                        models_best[model_type] = metric_[2:]
                        break
            models_best[baseline] = eval_metrics_base[np.argmax(eval_metrics_base[:, 1])][2:]
            for model_type in model_names_:
                if model_type not in models_best.keys():
                    eval_ = calculate_best_model(df[df['model'] == model_type])
                    models_best[model_type] = eval_[np.argmax(eval_[:, 1])][2:]
        elif metric == 'loss':
            best_loss = eval_metrics_base[np.argmin(eval_metrics_base[:, 0])][0]
            for model_type in model_names_:
                eval_metrics = calculate_best_model(df[df['model'] == model_type])
                for metric_ in eval_metrics:
                    if metric_[0] < best_loss:
                        models_best[model_type] = metric_[2:]
                        break
            models_best[baseline] = eval_metrics_base[np.argmin(eval_metrics_base[:, 0])][2:]
            for model_type in model_names_:
                if model_type not in models_best.keys():
                    eval_ = calculate_best_model(df[df['model'] == model_type])
                    models_best[model_type] = eval_[np.argmin(eval_[:, 0])][2:]
    else:
        models_best = {}
        for model_type in model_names:
            eval_ = calculate_best_model(df[df['model'] == model_type])
            if metric == 'loss':
                models_best[model_type] = eval_[np.argmin(eval_[:, 0])][2:]
            elif metric == 'accuracy':
                models_best[model_type] = eval_[np.argmax(eval_[:, 1])][2:]

    return models_best


def get_class_names(dataset, labels):
    if dataset == "mnist":
        class_names = {
            0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
            5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
        }
    elif dataset == "fmnist":
        class_names = {
            0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
            4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag",
            9: "Ankle boot",
        }
    elif dataset == "kmnist":
        class_names = {
            0: "お", 1: "き", 2: "す", 3: "つ", 4: "な",
            5: "は", 6: "ま", 7: "や", 8: "れ", 9: "を",
        }
    elif dataset == "emnist":
        class_names = {
            0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
            5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
            10: "A", 11: "B", 12: "C", 13: "D",
            14: "E", 15: "F", 16: "G", 17: "H",
            18: "I", 19: "J", 20: "K", 21: "L",
            22: "M", 23: "N", 24: "O", 25: "P",
            26: "Q", 27: "R", 28: "S", 29: "T",
            30: "U", 31: "V", 32: "W", 33: "X",
            34: "Y", 35: "Z", 36: "a", 37: "b",
            38: "d", 39: "e", 40: "f", 41: "g",
            42: "h", 43: "n", 44: "q", 45: "r",
            46: "t",
        }
    elif dataset == "cifar10":
        class_names = {
            0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",
        }
    else:
        raise ValueError("Invalid dataset. Supported datasets:"
                         "'mnist', 'fmnist', 'emnist', 'kmnist', 'cifar10'")

    # Define the get_class_name function
    def get_class_name(label):
        return class_names.get(label, "Unknown")

    # Use np.vectorize to apply the function to the entire array
    vectorized_get_class_name = np.vectorize(get_class_name)
    class_names_array = vectorized_get_class_name(labels)

    return class_names_array


def calculate_proj_scores(model_to_keep, dirname_figs, sigma=0.0,
                          dim_method="tsne", datatype="fmnist",
                          seq_tag="_sequential", learn_phase="trained",
                          n_trials=5, rep=1):
    df_scores = pd.DataFrame()
    for model_name in model_to_keep:
        postfix = f"{dim_method}_{datatype}{seq_tag}_sigma_{sigma}_{learn_phase}_rep_{rep}"
        fname_store = pathlib.Path(f"{dirname_figs}/post_analysis_embeddings_{postfix}")
        with open(f'{fname_store}.pkl', 'rb') as file:
            results = pickle.load(file)
        scores = results['scores']
        embeddings = results['embeddings']
        for t in range(1, n_trials + 1):
            for l in [2, 4, 5]:
                scores_ = scores[model_name][f'trial_{t}'][f'layer_{l}']
                df_ = pd.DataFrame(index=[0])
                df_['model'] = model_name
                if l == 2:
                    df_['layer'] = 'dendritic'
                elif l == 4:
                    df_['layer'] = 'somatic'
                elif l == 5:
                    df_['layer'] = 'output'
                df_['silhouette'] = scores_[0]
                df_['nh_score'] = scores_[1]
                df_['trustworthiness'] = scores_[2]
                df_['trial'] = t
                df_['sigma'] = sigma
                df_scores = pd.concat([df_scores, df_], ignore_index=True)
    return df_scores, embeddings


def draw_text_metrics(ax, xloc, yloc, metric, text,
                      color='black', fontsize=9):
    ax.text(
        x=xloc,
        y=yloc,
        transform=ax.transAxes,
        s=f"{text}: {metric:.3f}",
        fontweight='demibold',
        fontsize=fontsize,
        verticalalignment='top',
        horizontalalignment='right',
        backgroundcolor='white',
        color=color
    )


def make_subplots(fig, fig_part, dataset="mnist", label="A"):

    # Load the dataset
    data, labels, img_height, img_width, channels = get_data(
        validation_split=0.1,
        dtype=dataset,
    )
    x_train = data['train']
    y_train = labels['train']
    x = x_train[0].reshape(img_width, img_height, channels).squeeze()

    # Make the mosaic based on the number of classes
    n_classes = len(set(y_train))

    if dataset == "emnist":
        mosaic = np.arange((int(n_classes / 10) + 1)*10).reshape(5, 10).astype('str')
    else:
        mosaic = np.arange(n_classes).reshape(2, 5).astype('str')

    axd = fig_part.subplot_mosaic(
        mosaic,
        gridspec_kw={
            "wspace": 0.0,
            "hspace": 0.0,
            "left": 0.0,
            },
        )

    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    axd["0"].text(
        0.0, 1.0, label,
        transform=axd["0"].transAxes + trans,
        fontsize='large', va='bottom'
        )

    for i, (labels, ax) in enumerate(axd.items()):
        # Remove x-axis and y-axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.grid(False)
        if i > n_classes - 1:
            continue
        x = x_train[y_train == i][0].reshape(img_width, img_height, channels).squeeze()
        isns.imshow(
            x,
            gray=True if channels == 1 else False,
            cbar=False,
            square=True,
            ax=ax,
            )


def short_to_long_names(names):

    long_names = []
    for n in names:
        if n == "dANN-R":
            long_names.append("dend_ann_random")
        elif n == "dANN-LRF":
            long_names.append("dend_ann_local_rfs")
        elif n == "dANN-GRF":
            long_names.append("dend_ann_global_rfs")
        elif n == "pdANN":
            long_names.append("dend_ann_all_to_all")
        elif n == "sANN":
            long_names.append("sparse_ann")
        elif n == "sANN-LRF":
            long_names.append("sparse_ann_local_rfs")
        elif n == "sANN-GRF":
            long_names.append("sparse_ann_global_rfs")
        elif n == "psANN":
            long_names.append("sparse_ann_all_to_all")
        elif n == "vANN-R":
            long_names.append("vanilla_ann_random")
        elif n == "vANN-LRF":
            long_names.append("vanilla_ann_local_rfs")
        elif n == "vANN-GRF":
            long_names.append("vanilla_ann_global_rfs")
        elif n == "vANN":
            long_names.append("vanilla_ann")
    return long_names


def calc_eff_scores(df, form='acc'):
    df_out = pd.DataFrame()
    datasets = df["data"].unique()
    for d in datasets:
        dataset_df = df[df["data"] == d].copy()

        factor_params = dataset_df["trainable_params"]
        factor_epochs = dataset_df["num_epochs_min"] + 1

        f = np.log10(factor_params*factor_epochs)
        f /= f.min()

        if form == "acc":
            dataset_df["normed_acc"] = (dataset_df["test_acc"] / 100) / f
            df_out = pd.concat([df_out, dataset_df], ignore_index=True)
        elif form == "loss":
            dataset_df["normed_loss"] = dataset_df["test_loss"] * f
            df_out = pd.concat([df_out, dataset_df], ignore_index=True)

    return df_out

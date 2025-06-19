#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:56:21 2023

@author: spiros
"""
import numpy as np
import os

if os.environ.get("CUDA_VISIBLE_DEVICES", "0") == "1":
    from cuml import set_global_output_type
    # from cuml.manifold import TSNE, UMAP
    from cuml.neighbors import NearestNeighbors

    set_global_output_type('numpy')
else:
    # from sklearn.manifold import TSNE, UMAP
    from sklearn.neighbors import NearestNeighbors

# from sklearn.decomposition import PCA


def calculate_sparsity(arr, threshold=0):
    """
    Calculate the sparsity of a layer

    Parameters
    ----------
    arr : np.ndarray
        Activations of a layer in `(N, D)` shape, where `N` is the number of inputs
        and `D` the number of nodes in that layer.
    threshold : float, optional
        Threshold above which a node is considered to be active. The default is 0.

    Returns
    -------
    sparsity : np.ndarray
        Sparsity per input in (N,) shape.

    """
    # Convert the output to a binary format
    binary_output = (arr > threshold).astype(int)
    # Calculate the sparsity
    sparsity = np.count_nonzero(binary_output == 0, axis=1) / binary_output.shape[1]
    return sparsity


def compute_hit_matrix(arr, y, threshold=0.0):
    """
    Computes the hit matrix for a given layer of an ANN model,
    using a matrix of layer activations.

    Args:
        arr (numpy.ndarray):
            A matrix of layer activations, with one row per input sample
            and one column per node.
        y (numpy.ndarray):
            The target labels, represented as an array of integers.
        threshold (float):
            The threshold above which a node considered to be active.
            Default is 0.

    Returns:
        numpy.ndarray:
            The hit matrix for the layer.
    """
    # Compute the hit matrix for the layer
    y = y.squeeze()
    num_classes = np.unique(y).shape[0]
    hit_matrix = np.zeros((num_classes, arr.shape[1]))
    for i in range(num_classes):
        hit_matrix[i] = np.sum(arr[y == i] > threshold, axis=0)
    return hit_matrix


def calc_inactive_nodes(hit_matrix, theta=0):
    return np.where(hit_matrix.sum(axis=0) <= theta)[0]


def zero_div(x, y):
    """
    Handle division with zero.

    Parameters
    ----------
    x : numpy.ndarray
        The numerator.
    y : numpy.ndarray
        The denominator.

    Returns
    -------
    float
        Returns zero if any element of `y` is zero, otherwise returns `x / y`.

    """
    return 0.0 if y.any() == 0 else x / y


def add_countless(hit_matrix, total_inputs):
    countless = (len(total_inputs) - hit_matrix.sum(axis=0)).reshape(1, -1)
    return np.concatenate((hit_matrix, countless), axis=0)


def compute_entropy(arr):
    """
    Compute the entropy of an array `arr`.

    Parameters
    ----------
    arr : numpy.ndarray
        The array with probabilities.

    Returns
    -------
    float
        The entropy H[x] = -\sum p(x) log2 p(x).

    """
    # make an array,  if `arr` is list.
    arr = np.array(arr).astype('float64')
    # normalize probabilities to sum to 1 if they don't.
    p = arr/np.sum(arr) if np.sum(arr, axis=0).any() != 1 else arr
    # calculate log and set log(0) to 0.
    logp = np.log2(p, out=np.zeros_like(arr), where=(arr!=0))
    return -np.sum(p * logp, axis=0)


def compute_node_entropies(hit_matrix, true_labels):
    """
    Computes the entropy of each node per class using the hit matrix.

    Args:
        hit_matrix (numpy.ndarray):
            The hit matrix, with one row per class and one column per node.
        total_inputs (numpy.ndarray)
            The total number of inputs for which the entropy is calculated.

    Returns:
        entropies (numpy.ndarray):
            The entropy per node, i.e, low values show class specific behavior,
            whereas large values show mixture selectivity.
    """
    # Compute keep the nodes that are not silent
    hit_matrix = add_countless(hit_matrix, true_labels)
    probabilities = hit_matrix / hit_matrix.sum(axis=0)
    if not probabilities.sum(axis=0).all() == 1.:
        raise ValueError("Node probabilities must sum to 1.")

    # Compute the entropy of the probability distribution for each node and each class
    # calculate the entropy by removing the last row
    entropies = compute_entropy(probabilities)

    if sum(entropies < 0) != 0:
        raise ValueError("Entropy can't be negative!")
    return entropies


def node_specificity(hit_matrix, theta=0):
    """
    The selectivity index.

    Parameters
    ----------
    hit_matrix : numpy.ndarray
        The hit matrix with counts per class and nodes.
    theta : int, optional
        The threshold above which a node is active for a class. The default is 0.

    Returns
    -------
    numpy.ndarray
        An array equal to number of nodes with values in [0, `nclasses`].

    """
    return np.sum(hit_matrix > theta, axis=0)


def information_metrics(activation_arr, true_labels, theta=0,):
    # Information theory metrics
    hit_matrix = compute_hit_matrix(activation_arr, true_labels)
    inactive_nodes = calc_inactive_nodes(hit_matrix, theta=theta)
    entropy = np.delete(compute_node_entropies(hit_matrix, true_labels), inactive_nodes)
    sparsity = len(inactive_nodes)/hit_matrix.shape[1]

    # Calculate the node selectivity
    selectivity = node_specificity(hit_matrix, theta=400)
    return entropy, sparsity, inactive_nodes, selectivity


# def _dim_reduction(arr, k=2, method='umap',
#                    pca_preprocess=False,
#                    device='gpu', seed=None):
#     # Perform t-SNE visualization
#     set_global_device_type(device)
#     if method == 'tsne':
#         model = TSNE(
#             n_components=k,
#             perplexity=50.0,
#             n_neighbors=3*50.0,
#             random_state=seed)
#     elif method == 'umap':
#         model = UMAP(
#             n_components=k,
#             min_dist=0.0,
#             n_neighbors=30,
#             random_state=seed)
#     elif method == 'pca':
#         model = PCA(
#             n_components=k
#             )
#     if pca_preprocess:
#         # dim reduction
#         pca_model = PCA(n_components=50)
#         arr = pca_model.fit_transform(arr)
#     model_result = model.fit_transform(arr)
#     return model_result


def neighborhood_hit(arr, labels, k=5, metric='minkowski'):
    # find the kn indices with lowest distance metric.
    model = NearestNeighbors(n_neighbors=k, metric=metric).fit(arr)
    distances, indices = model.kneighbors(arr)
    if indices.shape[1] != k:
        raise FloatingPointError(f"More/less neighbors found. They should be {k}")
    neighbors_class = labels[indices]
    return np.mean(np.sum(neighbors_class == np.tile(np.expand_dims(labels, axis=1), k), axis=1) / k)


def num_trainable_params(D, S, n_input=28*28, n_syn=16, n_class=10,
                         num_layers=1, model_type='vanilla_ann'):
    """
    Calculate the actual number of trainable parameters.

    Parameters
    ----------
    D : int
        Number of dendrites per soma.
    S : int
        Number of somata.
    n_input : int, optional
        Number of inputs. The default is `28*28`.
    n_syn : int, optional
        Number of synapses, i.e., inputs per dendrites. The default is 16.
    n_class : int, optional
        Number of classes in the dataset. The default is 10.
    num_layers : int
        Number of effective layers
        (dendrosomatic layers, so 1 means 2 hidden layers). Default is 1.
    model_type : str, optional
        The type of model. The default is 'vanilla_ann'.

    Returns
    -------
    int
        Number of trainable parameters.

    """
    if model_type == 'vanilla_ann' or 'vanilla_ann_dropout_' in model_type:
        # fully connected layers.
        return ((n_input+1)*D*S + (D*S+1)*S)*num_layers + (S+1)*n_class
    elif model_type == 'vanilla_ann_local_rfs' or model_type == 'vanilla_ann_global_rfs' or model_type == 'vanilla_ann_random':
        # input is sparse, then is fully connected.
        return ((n_syn+1)*D*S + (D*S+1)*S)*num_layers + (S+1)*n_class
    elif model_type == 'dend_ann_all_to_all' or model_type == 'sparse_ann_all_to_all':
        # fully connected then sparse
        return ((n_input+1)*D*S + (D+1)*S)*num_layers + (S+1)*n_class
    else:
        # all dendritic and sparse models.
        return ((n_syn+1)*D*S + (D+1)*S)*num_layers + (S+1)*n_class


def get_power_of_10(number):
    """
    Return the power of 10 of a given `number`.

    Parameters
    ----------
    number : int
        A positive number.

    Returns
    -------
    int
        The closest appriximation of that number in power of 10.

    """
    return int(np.round(np.log10(number)))


def remove_zeros(arr):
    """
    Removes the zero entries from an array.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array.

    Returns
    -------
    numpy.ndarray
        Returns an array with less or equal size to `arr`.

    """
    return (arr[arr != 0])


def get_layer_weights(model, layer_name='dend_1'):
    for layer in model.layers:
        if layer.name.__contains__(layer_name):
            return layer.weights[0].numpy()

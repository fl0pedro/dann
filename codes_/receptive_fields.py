#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:39:08 2020.

@author: spiros and florian
"""
import numpy as np
import scipy.ndimage as scn
from typing import Literal, Callable
from numpy.typing import NDArray

# TODO: come up with a better fn name
# TODO: this isn't used to its all capbaility, i.e., not efficient.
def nb_vals(shape: tuple[int, int], indices: list[tuple[int, int]], size: int=1, opt: Literal["extend"] | None = None) -> list[tuple[int, int]]:
    """
    https://stackoverflow.com/questions/49210506/how-to-get-all-the-values-of-neighbours-around-an-element-in-matrix.

    Parameters
    ----------
    matrix : the input numpy.array.
    indices : list with the center of the neighborhood.

    Returns
    -------
    Array : flat indices of the neighborhood.
    
    """
    dist = np.ones(shape)
    dist[indices] = 0
    dist = scn.distance_transform_cdt(dist)

    if opt == "extend":
        nb_mask = dist == size
    else:
        nb_mask = dist <= size
    
    return np.nonzero(nb_mask)

def one_to_one_idxs(inputs: int, outputs: int, rng) -> list[tuple[int, int]]:
    # TODO docs and typing for rng
    if inputs < outputs:
        idxs = rng.integers(0, inputs, outputs)
    else:
        idxs = rng.choice(inputs, outputs)

    return [idxs[i]*i for i in range(outputs)]

def random_idxs(inputs: int, outputs: int, conns: int, rng):
    # TODO docs and typing for rng
    """
    """
    size = inputs*outputs
    
    if conns is None or conns <= 0 or not isinstance(conns, int):
        raise ValueError("Specify `conns` as positive integer.")
    if conns > size: 
        raise ValueError("`conns` must be lower than `size=input*output`")

    return rng.choice(size, conns, replace=False)

def constant_idxs(inputs: int, outputs: int, conns: int, rng):
    # TODO docs and typing for rng
    
    if conns is None or conns <= 0 or not isinstance(conns, int):
        raise ValueError("Specify `conns` as positive integer.")
    if conns > inputs:
        raise ValueError("`conns` must be lower than `input`")

    return [
        rng.choice(inputs, conns, replace=False)*i
        for i in range(outputs)
    ]

def random_connectivity(inputs: int, outputs: int, opt: Literal["one_to_one", "random", "constant"] = "random", conns: int = None, seed: int = None):
    # TODO docs and determine use of seed or rng
    rng = np.random.default_rng(seed)

    if opt == "one_to_one":
        idxs = one_to_one_idxs(inputs, outputs, rng)
    elif opt == "random":
        idxs = random_idxs(inputs, outputs, conns, rng)
    elif opt == "constant":
        idxs = constant_idxs(inputs, outputs, conns, rng)
    else:
        raise ValueError("Not a valid option. `opt` should take the values`one_to_one`, `random` or `constant`")
    
    mask = np.zeros(shape=(inputs, outputs))
    mask.flat[idxs] = 1
    return mask.astype('int')

def allocate_synapses(shape, nb, num_of_synapses, num_channels=1, seed=None, flat=True):
    # TODO docs and determine use of seed
    rng = np.random.default_rng(seed)

    mask = np.zeros(shape)
    
    x_idxs = []
    y_idxs = []
    length = 0

    for i in range(max(shape)): # do while (can't be greater than max shape) 
        x, y = nb_vals(mask.shape, nb, size=i, opt="extend")

        x_idxs.extend(x)
        y_idxs.extend(y)
        length += len(x)
        if length >= num_of_synapses: # condition for the do while
            break

    synapse_idxs = (x_idxs, y_idxs)
   
    if length > num_of_synapses:
        synapse_idxs = tuple(rng.choice(synapse_idxs, num_of_synapses, replace=False, axis=1))
    
    mask[synapse_idxs] = 1
    
    if num_channels > 1:
        mask = np.tile(mask[..., np.newaxis], num_channels)
    
    return mask.reshape(-1) if flat else mask
    


def make_mask_matrix(
    center_idxs, shape, dendrites, somata,
    num_of_synapses, num_channels=1,
    rfs_type='somatic', seed=None
    ):
    # TODO: docs and determine use of seed
    # TODO: rng is kind of out of place here, it would be nice to fix nb_vals, and allocate_synapses such that it finds the correct neighbors for each in one go.
    rng = np.random.default_rng(seed)
    # shouldn't the shape be (num_of_synapses[*dendrites], ...) where [] is for ref_type="somatic"
    size = np.prod(shape)
    mask = np.zeros((dendrites*somata, size*num_channels))

    if rfs_type == "somatic":
        # for center in center_idxs: ... # serial solution, guarantees each soma has the same number of dendrites.
        somatic_mask = allocate_synapses(shape, center_idxs, somata*dendrites, num_channels, flat=False)
        center_idxs = np.nonzero(somatic_mask)

    if size < dendrites*somata: # not enough centers
        center_idxs = rng.choice(center_idxs, dendrites*somata, axis=1)
   
    for i, center in enumerate(np.transpose(center_idxs)):
        mask[i, :] = allocate_synapses(shape, center, num_of_synapses, num_channels)

    return mask

def binary_uniform(max_val: int, size: int, shape: tuple[int, int], split: float, prob: float, rng):
    # TODO: finish (not used though)
    # TODO: docs
    res = np.empty(size)
    p = rng.random(size)
    res[p > prob]
    somata1 = sum(p > prob)  # outside of attention site
    somata2 = sum(p < prob)  # inside attention site

    # image center coordinates
    h, w = shape
    # this looks wrong
    w1, w2 = h//2 - h//4, h//2 + h//4
    h1, h2 = w//2 - w//4, w//2 + w//4
    # Random allocation outside of the middle of the image
    centers_w = rng.choice(
        list(range(w1)) + list(range(w2, w)),
        somata1,
    )
    centers_h = rng.choice(
        list(range(h1)) + list(range(h2, h)),
        somata1,
    )
    centers_ids1 = list(zip(centers_w, centers_h))

    # Random allocation in the middle of the image
    centers_w = rng.choice(range(w1, w2), somata2)
    centers_h = rng.choice(range(h1, h2), somata2)
    centers_ids2 = list(zip(centers_w, centers_h))

    if centers_ids1 is not None and centers_ids2 is not None:
        centers_ids = centers_ids1 + centers_ids2
    elif centers_ids1 is None and centers_ids2 is not None:
        centers_ids = centers_ids2
    elif centers_ids1 is not None and centers_ids2 is None:
        centers_ids = centers_ids1

    return center_idxs

# bbox = (x, y, len_x, len_y)
def generate_idxs(fn: Callable[(int, int), NDArray], shape, rng, fn_kws: dict | tuple[dict, dict] = None, size=1, bbox: tuple[int, int, int, int] | None = None):
    # TODO docs
    max_x, max_y = shape

    if fn_kws is None:
        fn_kws = {}

    if bbox is None:
        bbox = (0, 0, max_x, max_y)
    
    assert all(x >= 0 for x in bbox), "All entries of the bounding box must be positive"
    x, y = min(bbox[0], max_x), min(bbox[1], max_y)
    len_x, len_y = min(bbox[2], max_x-x), min(bbox[3], max_y-y)

    # NOTE: fn must handle value 0 and return intiger arrays!
    
    if isinstance(fn_kws, tuple):
        center_idxs = (x + fn(len_x, size, **fn_kws[0]), y + fn(len_y, size, **fn_kws[1]))
    else:
        flat_bbox_idxs = fn(len_x*len_y, size, **fn_kws)
        center_idxs = (x + flat_bbox_idxs%len_x, y + flat_bbox_idxs//len_x)
    
    return center_idxs 

def receptive_fields(shape, somata, dendrites, synapses, fn, typ="somatic", num_channels=1, rng=None, center_kws=None, center_idxs=None):
    # TODO: docs
    if center_kws is None:
        center_kws = {}
    
    if typ == "somatic":
        nodes = somata
    elif typ == "dendritic":
        nodes = dendrites*somata
    
    if center_idxs is None:
        center_idxs = generate_idxs(fn, shape, rng, size=nodes, **center_kws)
   
    mask_final = make_mask_matrix(center_idxs, shape, dendrites, somata, synapses, num_channels, typ, rng)

    return mask_final.astype("int").T, center_idxs


def connectivity(inputs, outputs):
    # TODO docs
    if outputs <= 0:
        raise ValueError("Number of outputs must be greater than zero.")
    if inputs <= 0:
        raise ValueError("Number of inputs must be greater than zero.")
    if inputs % outputs != 0:
        raise ValueError("Inputs must be divisible by outputs without a remainder.")

    connectivity_matrix = np.zeros((inputs, outputs), dtype=int)
    in_per_out = inputs // outputs  # nodes per node
    # Fill the connectivity matrix
    for j in range(outputs):
        start_index = in_per_out * j
        end_index = start_index + in_per_out
        connectivity_matrix[start_index:end_index, j] = 1
    return connectivity_matrix


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:39:08 2020.

@author: spiros and florian
"""
import numpy as np
import scipy.ndimage as scn
from typing import Literal, Callable
from numpy.typing import NDArray, ArrayLike
from numpy.random import Generator

# TODO: the two connectivities and receptive fields need to accept rng and seeds too.

# TODO: come up with a better fn name
# TODO: this isn't used to its all capbaility, i.e., not efficient.
def neighbors(shape: tuple[int, int], indices: tuple[ArrayLike, ArrayLike], size: int = 1, boundary: bool = False) -> tuple[NDArray[int], NDArray[int]]:
    """
    Finds the neighbors around several centroids.
    https://stackoverflow.com/questions/49210506/how-to-get-all-the-values-of-neighbours-around-an-element-in-matrix.

    Parameters
    ----------
    shape : the array shape.
    indices : list of centroids.
    size : (_optional_) size of the neighborhood, default is `1`.
    boundary : (_optional_) whether to return only the boundary, default is `False`.
    
    Returns
    -------
    Array : indices of the neighborhood.
    
    """
    dist = np.ones(shape)
    dist[indices] = 0
    dist = scn.distance_transform_cdt(dist)

    if boundary: 
        nb_mask = dist == size 
    else:
        nb_mask = dist <= size
    
    return np.nonzero(nb_mask)

def one_to_one_idxs(inputs: int, outputs: int, rng: Generator) -> NDArray[int]:
    """
    Parameters
    ----------
    inputs : input size.
    outputs : output size.
    rng : numpy random number generator.

    Returns
    -------
    Array : flat indices.

    """
    out_idxs = rng.choice(inputs, outputs, replace=inputs<outputs)

    return np.arange(outputs)+out_idxs*outputs

def random_idxs(inputs: int, outputs: int, conns: int, rng: Generator) -> NDArray[int]:
    """
    Parameters
    ----------
    inputs : input size.
    outputs : output size.
    conns : number of connections, must be a value between zero and `size=inputs*output`.
    rng : numpy random number generator.

    Returns
    -------
    Array : flat indices.

    """
    size = inputs*outputs
    
    if conns is None or conns <= 0 or not isinstance(conns, int):
        raise ValueError("Specify `conns` as positive integer.")
    if conns > size: 
        raise ValueError("`conns` must be lower than `size=input*output`")

    return rng.choice(size, conns, replace=False)

def constant_idxs(inputs: int, outputs: int, conns: int, rng: Generator) -> NDArray[int]:
    """
    Each output will get some random inputs.

    Parameters
    ----------
    inputs : input size.
    outputs : output size.
    conns : number of connections, must be a value between zero and `input`.
    rng : numpy random number generator.
    
    Returns
    -------
    Array : flat indices.

    """
    if conns is None or conns <= 0 or not isinstance(conns, int):
        raise ValueError("Specify `conns` as positive integer.")
    if conns > inputs:
        raise ValueError("`conns` must be lower than `input`")

    return np.array([
        i+x*outputs
        for i in range(outputs)
        for x in rng.choice(inputs, conns, replace=False)
    ])

def random_connectivity(inputs: int, outputs: int, rng: Generator, opt: Literal["one_to_one", "random", "constant"] = "random", conns: int | None = None) -> NDArray[int]:
    """
    Generates a connectivity with random connections. The structure is randomly generated depending on the `opt`.
    The connectivity is a projection, from the input to some output, restrictions depend on `opt`.

    Parameters
    ----------
    inputs : input size.
    outputs : output size.
    rng : numpy random number generator.
    opt : (_optional_) option for randomness: `one_to_one`, `random`, or `constant`, default is `random`.
    conns : (_optional_) number of connections.

    Returns
    -------
    Array : connectivity network.

    """
    if opt == "one_to_one":
        idxs = one_to_one_idxs(inputs, outputs, rng)
    elif opt == "random":
        idxs = random_idxs(inputs, outputs, conns, rng)
    elif opt == "constant":
        idxs = constant_idxs(inputs, outputs, conns, rng)
    else:
        raise ValueError("Not a valid option. `opt` should take the values `one_to_one`, `random` or `constant`")
    
    connectivity_matrix = np.zeros((inputs, outputs))
    connectivity_matrix.flat[idxs] = 1
    return connectivity_matrix.astype('int')

def structured_connectivity(inputs: int, outputs: int) -> NDArray[int]:
    """
    Generates a connectivity matrix with local connections. The structure is block sparse, with the width of each
    individual block being one. The connectivity is a projection from an input that is divisible by some output.

    Parameters
    ----------
    inputs : input size, must be greater than zero and divisible by `outputs`.
    outputs : output size, must be greater than zero.

    Returns
    -------
    Array : connectivity network.

    """
    if outputs <= 0:
        raise ValueError("Number of outputs must be greater than zero.")
    if inputs <= 0:
        raise ValueError("Number of inputs must be greater than zero.")
    if inputs % outputs != 0:
        raise ValueError("Inputs must be divisible by outputs without a remainder.")

    connectivity_matrix = np.zeros((inputs, outputs), dtype=int)
    conns = inputs // outputs
    for j in range(outputs):
        i = j*conns
        connectivity_matrix[i:i+conns, j] = 1
    
    return connectivity_matrix

def allocate_synapses(shape, nb, num_of_synapses, rng, num_channels=1, flat=True):
    """
    
    """
    mask = np.zeros(shape)
    
    x_idxs = []
    y_idxs = []
    length = 0

    for i in range(max(shape)): # like a do while, but stops after maximum iteration (shape size) 
        x, y = neighbors(mask.shape, nb, size=i, boundary=True)

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

def new_neighbors(shape, center_idxs, size, rng): # return an array of shape (len(center_idxs),2,size)
    nb_len = math.ceil(math.sqrt(size))
    w, h = shape
    for x, y in center_idxs:
        if any(l >= z for z in {w-x, x, h-y, y}):
            dist = scn.distance_transform_cdt(dist)
            count = 0
            for l in range(max(shape)):
                count += sum(dist == l)
                if count >= size:
                    break
        else:
            l = nb_len
        
        min_x, min_y = max(0, x-l), max(0, y-l)
        len_x, len_y = min(w-x, l), min(h-y, l)

        if size > nb_len**2:
            center_idxs = (min_x + rng.choice(len_x, replace=False), min_y + rng.choice(len_y, replace=False))
        else:
            center_idxs = (min_x + np.arange(len_x), min_y + np.arange(len_y))


def make_mask_matrix(center_idxs, shape, dendrites, somata, num_of_synapses, num_channels=1, rfs_type='somatic', seed: int | None = None):
    # TODO: docs and determine use of seed
    # TODO: rng is kind of out of place here, it would be nice to fix `neighbors`, and allocate_synapses such that it finds the correct neighbors for each in one go.
    rng = np.random.default_rng(seed)
    # shouldn't the shape be (num_of_synapses[*dendrites], ...) where [] is for ref_type="somatic"
    size = np.prod(shape)
    mask = np.zeros((dendrites*somata, size*num_channels))

    if rfs_type == "somatic":
        # for center in center_idxs: ... # serial solution, guarantees each soma has the same number of dendrites.
        somatic_mask = allocate_synapses(shape, center_idxs, somata*dendrites, rng, num_channels)
        center_idxs = np.nonzero(somatic_mask)

    if size < dendrites*somata: # not enough centers
        center_idxs = rng.choice(center_idxs, dendrites*somata, axis=1)
   
    for i, center in enumerate(np.transpose(center_idxs)):
        mask[i, :] = allocate_synapses(shape, center, num_of_synapses, rng, num_channels)

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


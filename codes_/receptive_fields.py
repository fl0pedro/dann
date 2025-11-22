#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from functools import partial
from typing import Literal
from math import isqrt

def squareish_shape(n):
    limit = isqrt(n)
    for h in range(limit, 0, -1):
        if n % h == 0:
            w = n // h
            return (h, w)
    return (1, n)

def get_coords(shape: tuple[int, int]) -> tuple[jnp.ndarray, jnp.ndarray]:
    h, w, *_ = shape
    return jnp.meshgrid(jnp.arange(w), jnp.arange(h))

def get_masks(key: jnp.ndarray, config) -> list[jnp.ndarray]:
    if not config.original:
        raise ValueError("Config must be set to original mode.")
    
    masks_list = []
    
    for i in range(config.layers):
        key, k1, k2 = jr.split(key, 3)
        
        # input
        if i == 0:
            shape = config.input_shape[1:]
            input_size = config.input_size
            num_channels = config.input_shape[0]
        else:
            shape = squareish_shape(config.soma[i-1])
            input_size = config.soma[i-1]
            num_channels = 1

        # synapse -> dendrite
        if config.rfs:
            mask_s_d, _ = receptive_fields(
                k1, shape=shape, somata=config.soma[i], dendrites=config.dends[i],
                synapses=config.nsyns, typ=config.rfs, num_channels=num_channels
            )
        else:
            total_inputs = input_size * num_channels
            total_units = config.soma[i] * config.dends[i]
            
            mask_s_d = random_connectivity(
                k1, inputs=total_inputs, outputs=total_units,
                opt="random", conns=config.nsyns * total_units
            )

        masks_list.append(mask_s_d)

        # dendrite -> soma
        total_dends = config.dends[i] * config.soma[i]
        
        if not config.sparse:
            mask_d_s = structured_connectivity(
                inputs=total_dends, outputs=config.soma[i]
            )
        else:
            mask_d_s = random_connectivity(
                k2, inputs=total_dends, outputs=config.soma[i],
                opt="random", conns=total_dends
            )
            
        masks_list.append(mask_d_s)

    # conditionally remove masks
    if config.conventional:
        if config.rfs or config.sparse: # dend-soma is still block diag
            for i, m in enumerate(masks_list):
                if i % 2 != 0:
                    masks_list[i] = jnp.ones_like(m)
        else:
            for i, m in enumerate(masks_list):
                masks_list[i] = jnp.ones_like(m)

    if config.all_to_all: # pdANN
        for i, m in enumerate(masks_list):
            if i % 2 == 0: 
                masks_list[i] = jnp.ones_like(m)

    return masks_list

def random_connectivity(
    key: jnp.ndarray,
    inputs: int,
    outputs: int,
    opt: Literal["one_to_one", "random", "constant"] = "random",
    conns: int | None = None
) -> jnp.ndarray:
    
    matrix = jnp.zeros((inputs, outputs), dtype=jnp.int32)
    
    if opt == "one_to_one":
        p = jr.permutation(key, jnp.arange(inputs))[:outputs]
        row_idx = p
        col_idx = jnp.arange(outputs)
        return matrix.at[row_idx, col_idx].set(1)

    elif opt == "random":
        if conns is None: 
            raise ValueError("conns required")
        flat_idx = jr.choice(key, inputs * outputs, shape=(conns,), replace=False)
        return matrix.flatten().at[flat_idx].set(1).reshape(inputs, outputs)

    elif opt == "constant":
        if conns is None:
            raise ValueError("conns required")
        
        def sample_col(k):
            idx = jr.choice(k, inputs, shape=(conns,), replace=False)
            col = jnp.zeros((inputs,), dtype=jnp.int32)
            return col.at[idx].set(1)
            
        keys = jr.split(key, outputs)
        return vmap(sample_col)(keys).T # (Inputs, Outputs)

    else:
        raise ValueError(f"Unknown opt: {opt}")

def binary_uniform(
    key: jnp.ndarray, 
    shape: tuple[int, int], 
    size: int, 
    split: float = 0.5, 
    prob: float = 0.5
) -> jnp.ndarray:
    H, W, *_ = shape
    
    y_grid, x_grid = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    
    cent_h, cent_w = H * split, W * split
    margin_h, margin_w = (H - cent_h) / 2.0, (W - cent_w) / 2.0
    
    center_mask = (
        (y_grid >= margin_h) & (y_grid < (H - margin_h)) &
        (x_grid >= margin_w) & (x_grid < (W - margin_w))
    )
    
    area_in = jnp.sum(center_mask)
    area_out = jnp.maximum(1.0, (H * W) - area_in)
    area_in = jnp.maximum(1.0, area_in)
    
    w_in = prob / area_in
    w_out = (1.0 - prob) / area_out
    
    weights = jnp.where(center_mask, w_in, w_out)
    
    weights = weights / jnp.sum(weights)
    
    flat_indices = jr.choice(
        key, H * W, shape=(size,), p=weights.flatten(), replace=False
    )
    
    coords_x = flat_indices % W
    coords_y = flat_indices // W
    
    return jnp.stack([coords_x, coords_y], axis=1)

@partial(jax.jit, static_argnames=['inputs', 'outputs'])
def structured_connectivity(inputs: int, outputs: int) -> jnp.ndarray:
    if inputs % outputs != 0:
        raise ValueError("inputs must be divisible by outputs")
    
    block_h = inputs // outputs
    identity = jnp.eye(outputs, dtype=jnp.int32)
    column_block = jnp.ones((block_h, 1), dtype=jnp.int32)
    return jnp.kron(identity, column_block)

def allocate_synapses(
    key: jnp.ndarray,
    shape: tuple[int, int],
    centers: jnp.ndarray, 
    num_synapses: int,
    num_channels: int = 1,
    flat: bool = True
) -> jnp.ndarray:
    H, W, *_ = shape
    N_centers = centers.shape[0]
    
    grid_x, grid_y = get_coords(shape)
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    diff_x = jnp.abs(grid_x[None, :] - centers[:, 0:1])
    diff_y = jnp.abs(grid_y[None, :] - centers[:, 1:2])
    dist = jnp.maximum(diff_x, diff_y)

    noise = jr.uniform(key, dist.shape) * 0.1
    _, top_indices = lax.top_k(-(dist + noise), num_synapses[0])

    mask_flat = jnp.zeros((N_centers, H * W), dtype=jnp.int32)
    
    row_indices = jnp.arange(N_centers)[:, None]
    mask_flat = mask_flat.at[row_indices, top_indices].set(1)

    if num_channels > 1:
        mask_flat = jnp.tile(mask_flat[..., None], (1, 1, num_channels))
        if flat:
             return mask_flat.reshape(N_centers, -1)
    
    return mask_flat if flat else mask_flat.reshape(N_centers, H, W)

def receptive_fields(
    key: jnp.ndarray,
    shape: tuple[int, int], 
    somata: int, 
    dendrites: int, 
    synapses: int, 
    typ: Literal["somatic", "dendritic"] = "somatic", 
    num_channels: int = 1
) -> tuple[jnp.ndarray, jnp.ndarray]:
    k1, k2 = jr.split(key)
    H, W, *_ = shape
    total_nodes = somata * dendrites if typ == "dendritic" else somata
    
    flat_locs = jr.choice(k1, H * W, shape=(total_nodes,), replace=True)
    center_x = flat_locs % W
    center_y = flat_locs // W
    centers = jnp.stack([center_x, center_y], axis=1)

    final_centers = centers
    if typ == "somatic" and dendrites > 1:
         final_centers = jnp.repeat(centers, dendrites, axis=0)
         
    mask = allocate_synapses(
        k2, shape, final_centers, synapses, num_channels=num_channels, flat=True
    )

    return mask.T, final_centers

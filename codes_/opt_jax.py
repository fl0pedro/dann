#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import prod, isqrt
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, lax
from tqdm import tqdm
import optax
from functools import partial, reduce
from itertools import product
from jax.nn import leaky_relu
from operator import mul, add
from jax.nn.initializers import he_normal
from receptive_fields import random_connectivity, receptive_fields, structured_connectivity
from typing import Callable, NamedTuple

def sync_model(config, dataset_info):
    update_dends_and_soma(config)
    update_model_id_and_name(config)
    update_dnn_values(config)
    update_dataset_values(config, dataset_info)
    return unique_model_name(config)

def unique_model_name(config):
    # prefix
    unique_vals = [config.model_name, config.dataset, str(config.layers)]

    # default 1e-3
    if config.lr != 1e-3:
        unique_vals += [f"lr_{config.lr}"]

    # default False
    if config.seq: 
        unique_vals += ["seq"]

    # default 0.0
    if config.drop_rate: 
        unique_vals += [f"dropout_{config.drop_rate}"]

    # postfix
    unique_vals += [f"noise_{config.sigma}", str(config.trial)]
    return "_".join(unique_vals)

def update_dends_and_soma(config):
    if len(config.dends) == 1 and config.layers > 1:
        config.dends *= config.layers # constant size of dends

    if len(config.soma) == 1 and config.layers > 1:
        config.soma *= config.layers

    if len(config.dends) != len(config.soma):
        raise ValueError("Number of dendrites and soma must be equal")

    config.layers = len(config.dends) # == len(config.soma)

def update_model_id_and_name(config): # make sure defaults don't override custom config
    if not (config.model_id is None) ^ (config.model_name is None): # xor
        raise ValueError("Must provide either model id or name, but not both")
    
    ordered_model_names = [
        'dend_ann_random', 'dend_ann_global_rfs', 'dend_ann_local_rfs', 
        'vanilla_ann', 'vanilla_ann_random', 'vanilla_ann_global_rfs', 'vanilla_ann_local_rfs', 
        'sparse_ann', 'sparse_ann_global_rfs', 'sparse_ann_local_rfs', 
        'dend_ann_all_to_all', 'sparse_ann_all_to_all', 'locally_connected'
    ]

    if config.model_id is None:
        config.model_id = ordered_model_names.index(config.model_name)
    else:
        config.model_name = ordered_model_names[config.model_id]

def update_dnn_values(config):
    # update param values to model defaults, keep user overrides (stored in backup)
    # this is probably not well implemented
    backup_keys = {"rfs", "conventional", "sparse", "all_to_all"}
    # default values are None or False, therefore bool(v) == False in these cases
    backup = {k: v for k, v in vars(config).items() if k in backup_keys and v}
    
    match config.model_id:
        case 0:
            pass
        case 1:
            config.rfs = "somatic"
        case 2:
            config.rfs = "dendritic"
        case 3:
            config.conventional = True
        case 4:
            config.conventional = True
            config.sparse = True
        case 5:
            config.conventional = True
            config.rfs = "somatic"
        case 6:
            config.conventional = True
            config.rfs = "dendritic"
        case 7:
            config.sparse = True
        case 8:
            config.rfs = "somatic"
            config.sparse = True
        case 9:
            config.rfs = "dendritic"
            config.sparse = True
        case 10:
            config.rfs = "somatic"
            config.all_to_all = True
        case 11:
            config.rfs = "somatic"
            config.sparse = True
            config.all_to_all = True
        case 12:
            ... # fully local conv network
        case 13:
            ... # local conv as rfs
        case _:
            ... # shouldn't be able to reach this
    
    for k in backup.keys():
        setattr(config, k, backup[k])

def update_dataset_values(config, dataset_info):
    config.input_shape = dataset_info.features["image"].shape
    config.input_size = prod(config.input_shape)
    config.classes = dataset_info.features["label"].num_classes

    if config.early_stop:
        config.epochs = 100
    else:
        config.epochs = {
            "mnist": 15,
            "fmnist": 25,
            "kmnist": 25,
            "emnist": 50,
            "cifar10": 50
        }[config.dataset]

    if config.seq:
        config.epochs *= 2

def dataloader(dataset, batch_size, data_dir, split):
    import tensorflow as tf
    tf.config.set_visible_devices([], device_type='GPU')

    import tensorflow_datasets as tfds

    tf_split = [
        f"train[:{split:.0%}]", # train
        f"train[{split:.0%}:]", # validation
        "test", # test
    ]

    loader, info = tfds.load(dataset, split=tf_split, batch_size=batch_size, data_dir=data_dir, as_supervised=True, with_info=True)
    loader = tfds.as_numpy(loader)
    loader = dict(zip(["train", "val", "test"], loader))
    
    return loader, info

# class Indices(NamedTuple):
#     sd: jnp.ndarray
#     ds: jnp.ndarray

class LayerParams(NamedTuple):
    sd_w: jnp.ndarray
    sd_b: jnp.ndarray
    ds_w: jnp.ndarray
    ds_b: jnp.ndarray
    # positions: Indices | None = None

class LayerMasks(NamedTuple):
    sd: jnp.ndarray
    ds: jnp.ndarray

class LayerOps(NamedTuple):
    sd: Callable
    ds: Callable

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

@jax.vmap
@partial(jax.vmap, in_axes=(0, None))
def get_indices_from_permutations(px, pys):
    return jnp.stack([pys[px], pys[px+1 % pys.size]])

# def get_indices(key, config):
#     indices_list = []
#     for i in range(config.layers):
#         key, k1, k2 = jr.split(key, 3)
#         
#         # input
#         if i == 0:
#             shape = config.input_shape[1:]
#             input_size = config.input_size
#             num_channels = config.input_shape[0]
#         else:
#             shape = squareish_shape(config.soma[i-1])
#             input_size = config.soma[i-1]
#             num_channels = 1
# 
#         # synapse -> dendrite
#         # (nsyns*dends, dends*soma)
#         if config.rfs:
#             ... # rf
#             permutations = ... # dends
#         else:
#             ... # random
#             permutations = ... # nsyn*dends
# 
#         indices = get_indices(*permutations.transpose(2, 0, 1))
#         indices_list.append(indices)
# 
#         # dendrite -> soma
#         # (dends*soma, soma)
#         total_dends = config.dends[i] * config.soma[i]
#         
#         if not config.sparse:
#             ... # block diagonal (no permutation)
#             permutations = jnp.arange(...).reshape(2, -1)
#         else:
#             ... # random
#             permutations = ... # both are size soma
# 
#         indices = get_indices_from_permutations(*permutations)
#         indices_list.append(indices)
# 
#     return indices_list


def linear(key, input_size, output_size, mask=None):
    if isinstance(input_size, tuple):
        input_size = prod(input_size)
    if isinstance(output_size, tuple):
        output_size = prod(output_size)
    weight = he_normal()(key, (input_size, output_size))
    bias = jnp.zeros(output_size)

    def apply(x, weight, bias):
        if x.ndim > 1:
            x = x.reshape(x.shape[0], -1)
        return x @ weight + bias

    return apply, weight, bias

def conv(key, input_shape, output_shape, kernel_shape, stride, local=False, padding="VALID", input_dilation=None, kernel_dilation=None):
    print(input_shape, output_shape, kernel_shape, stride)
    if local:
        fused_input = input_shape[2] * kernel_shape[0] * kernel_shape[1]
        w_shape = (output_shape[0], output_shape[1], fused_input, output_shape[2])
    else:
        w_shape = (*kernel_shape, input_shape[2], output_shape[2])

    weight = he_normal()(key, w_shape)
    bias = jnp.zeros((output_shape[2],)) 
    dimensions = ('NHWC', 'HWIO', 'NHWC')

    def apply(x, w, b):
        if local:
            out = lax.conv_general_dilated_local(
                x, w, stride, padding, kernel_shape, 
                input_dilation, kernel_dilation,
                dimensions
            )
        else:
            out = lax.conv_general_dilated(
                x, w, stride, padding,
                input_dilation, kernel_dilation,
                dimensions
            )
        return out + b

    return apply, weight, bias

def squareish_shape(n):
    limit = isqrt(n)
    for h in range(limit, 0, -1):
        if n % h == 0:
            w = n // h
            return (h, w)
    return (1, n)

def output_shape(input_shape, kernel_shape, stride):
    ih, iw, *_ = input_shape
    kh, kw = kernel_shape
    sh, sw = stride
    return ((ih-kh) // sh + 1), ((iw - kw) // sw + 1)

def best_stride(input_shape, kernel_shape, target):
    candidates = []
    for h, w, *_ in product(*[range(1, x+1) for x in input_shape]):
        oh, ow = output_shape(input_shape, kernel_shape, (h, w))
        candidates.append((abs(oh*ow - target), abs(h - w), h, w, oh*ow))

    _, _, sh, sw, out = min(candidates)
    return (sh, sw), out

def build_layer(key, config, i, current_shape):
    key, k1, k2 = jr.split(key, 3)
    
    nsyns = config.nsyns[i]
    dends = config.dends[i]
    soma = config.soma[i]

    if config.conventional or config.original: # fully connected
        # synapse/soma -> dendrite
        sd_f, sd_w, sd_b = linear(k1, current_shape, dends*soma)

        # dendrite -> soma
        ds_f, ds_w, ds_b = linear(k2, dends*soma, soma)
        
        current_shape = (soma,)

    elif config.rfs and config.local: # local rfs
        kernel_shape = squareish_shape(nsyns)
        
        if config.rfs == "dendritic":
            stride, intermediate_shape = best_stride(current_shape, kernel_shape, dends*soma)
            intermediate_shape = squareish_shape(intermediate_shape)
        else:
            stride = kernel_shape
            intermediate_shape = output_shape(current_shape, kernel_shape, stride)

        intermediate_shape = (*intermediate_shape, 1)

        # synapse/soma -> dendrite
        sd_f, sd_w, sd_b = conv(k1, current_shape, intermediate_shape, kernel_shape, stride, config.local)

        # dendrite -> soma
        if config.rfs == "dendritic":
            ds_f, ds_w, ds_b = linear(k2, prod(intermediate_shape), soma)
            current_shape = (soma,)
        else:
            dend_kernel = squareish_shape(dends)
            current_shape = output_shape(intermediate_shape, dend_kernel, stride)
            ds_f, ds_w, ds_b = conv(k2, intermediate_shape, (*current_shape, 1), dend_kernel, stride, config.local)

    elif config.rfs and not config.local: # non local rfs
        kernel_shape = squareish_shape(nsyns)
        
        if config.rfs == "dendritic":
            stride, _ = best_stride(current_shape, kernel_shape, dends)
            intermediate_shape = (1,)
        else:
            stride = kernel_shape
            intermediate_shape = (dends,)

        out_spatial = output_shape(current_shape, kernel_shape, stride)
        intermediate_shape += out_spatial
        
        # synapse/soma -> dendrite
        sd_f, sd_w, sd_b = conv(k1, current_shape, intermediate_shape, kernel_shape, stride, config.local, "SAME")
        
        # dendrite -> soma
        if config.rfs == "dendritic":
            ds_f, ds_w, ds_b = linear(k2, prod(intermediate_shape)*soma, soma)
            current_shape = (soma,)
        else:
            out_spatial_2 = output_shape(out_spatial, kernel_shape, stride)
            final_shape = (soma,) + out_spatial_2
            ds_f, ds_w, ds_b = conv(k2, intermediate_shape, final_shape, kernel_shape, stride, config.local, "SAME")
            current_shape = final_shape
    else:
        print(config.rfs, config.local, config.conventional)
        raise ValueError

    return key, LayerParams(sd_w, sd_b, ds_w, ds_b), LayerOps(sd_f, ds_f), current_shape

def model(config):
    params = []
    ops = []
    masks = []

    key, mask_key = jr.split(jr.PRNGKey(config.trial))
    
    if config.original:
        mask_list = get_masks(key, config)
    elif config.improved:
        index_list = get_indices(key, config)

    current_shape = config.input_shape

    for i in range(config.layers):
        key, layer_params, layer_ops, current_shape = build_layer(key, config, i, current_shape)
        ops.append(layer_ops)
        params.append(layer_params)
        if config.original:
            masks.append(LayerMasks(*mask_list[i*2:i*2+2]))

    print(*jax.tree.map(lambda x: x.shape, params), sep='\n')

    f_final, *final_params = linear(key, current_shape, config.classes)
    print(f"FinalParams(w={final_params[0].shape}, b={final_params[1].shape}")

    total_size = sum(
        getattr(p, a).size 
        for p in params 
        for a in ("sd_w", "sd_b", "ds_w", "ds_b")
    ) + final_params[0].size + final_params[1].size
    total_masked = sum(
        (getattr(p, m) == 0).sum() 
        for p in masks
        for m in ("sd", "ds") 
    )

    print(f"params={total_size}")
    if masks:
        print(f"masked={total_masked}\nactive={total_size-total_masked}")
    #exit()

    def predict(x, params, final_params, dropout_key=None):
        if dropout_key is not None:
            keys = jr.split(dropout_key, len(config.layers)*2)

        # kinda dumb :I would be nice if we preprocessed it to be this way :) 
        x = jnp.array(x, dtype=jnp.float32)

        for i, layer in enumerate(params):
            # synapse/soma -> dendrite
            sd_w = layer.sd_w*masks[i].sd if masks else layer.sd_w
            #x = x[None, ...]
            x = ops[i].sd(x, sd_w, layer.sd_b)
            if dropout_key is not None:
                keep = jr.bernoulli(keys[i], config.drop_rate, x.shape)
                x = jnp.where(keep,x / config.drop_rate, 0)
            x = leaky_relu(x, 0.1)

            # dendrite -> soma
            ds_w = layer.ds_w*masks[i].ds if masks else layer.ds_w
            #exit()
            x = ops[i].ds(x, ds_w, layer.ds_b)
            if dropout_key is not None:
                keep = jr.bernoulli(keys[i], config.drop_rate, x.shape)
                x = jnp.where(keep,x / config.drop_rate, 0)
            x = leaky_relu(x, 0.1)

        return f_final(x, *final_params)

    return predict, params, final_params

def sparse_categorical_accuracy_with_integer_labels(logits, labels):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels)

def inc_mean(mean, x, n):
    return x if mean is None else (mean * (n-1)  +  x)  /  n

def train_loop(key, model, params, dataloader, loss_fn, optimizer, batch_size, epochs, shuffle=None, early_stop=False):
    @jit
    def update(params, batch, opt_state):
        def loss_and_output(params):
            inputs, labels = batch
            outputs = model(inputs, *params)
            return loss_fn(outputs, labels).mean(), outputs

        (loss, outputs), grads = jax.value_and_grad(loss_and_output, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, outputs, params, opt_state

    model = jit(model)

    #exit()

    opt_state = optimizer.init(params)

    for epoch in range(epochs):
        local_loss = None
        local_acc = None
        for i, batch in (t:=tqdm(enumerate(dataloader["train"]), total=len(dataloader["train"]))):
            loss, outputs, params, opt_state = update(params, batch, opt_state)
            acc = sparse_categorical_accuracy_with_integer_labels(outputs, batch[1])

            local_loss = inc_mean(local_loss, loss, i+1)
            local_acc = inc_mean(local_acc, acc, i+1)
            t.set_description(f"{epoch=}, loss={local_loss:,.4f}, acc={local_acc:.2%}")

        if epoch % 5 == 4:
            local_loss = None
            local_acc = None
            for i, batch in (t:=tqdm(enumerate(dataloader["val"]), total=len(dataloader["val"]))):
                inputs, labels = batch
                outputs = model(inputs, *params)
                loss = loss_fn(outputs, labels).mean()
                acc = sparse_categorical_accuracy_with_integer_labels(outputs, labels)

                local_loss = inc_mean(local_loss, loss, i+1)
                local_acc = inc_mean(local_acc, acc, i+1)
                t.set_description(f"validation, loss={local_loss:,.4f}, acc={local_acc:.2%}")

    local_loss = None
    local_acc = None
    for i, batch in (t:=tqdm(enumerate(dataloader["test"]), total=len(dataloader["test"]))):
        inputs, labels = batch
        outputs = model(inputs, *params)
        loss = loss_fn(outputs, labels).mean()
        acc = sparse_categorical_accuracy_with_integer_labels(outputs, labels)

        local_loss = inc_mean(local_loss, loss, i+1)
        local_acc = inc_mean(local_acc, acc, i+1)
        t.set_description(f"test, loss={local_loss:,.4f}, acc={local_acc:.2%}")

    return params, outputs


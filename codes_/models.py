import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, jit
from functools import partial
from itertools import product
from jax.nn import leaky_relu
from jax.nn.initializers import he_normal
from typing import Callable, NamedTuple
from math import prod, isqrt, ceil
from receptive_fields import get_masks

class Indices(NamedTuple):
    sd: jnp.ndarray
    ds: jnp.ndarray

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

@jax.vmap
@partial(jax.vmap, in_axes=(0, None))
def get_indices_from_permutations(px, pys):
    return jnp.stack([pys[px], pys[px+1 % pys.size]])

def get_indices(key, config):
    indices_list = []
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
        # (nsyns*dends, dends*soma)
        if config.rfs:
            ... # rf
            permutations = ... # dends
        else:
            ... # random
            permutations = ... # nsyn*dends

        indices = get_indices(*permutations.transpose(2, 0, 1))
        indices_list.append(indices)

        # dendrite -> soma
        # (dends*soma, soma)
        total_dends = config.dends[i] * config.soma[i]
        
        if not config.sparse:
            ... # block diagonal (no permutation)
            permutations = jnp.arange(...).reshape(2, -1)
        else:
            ... # random
            permutations = ... # both are size soma

        indices = get_indices_from_permutations(*permutations)
        indices_list.append(indices)

    return indices_list


def linear(key, input_size, output_size):
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


def block_linear(key, input_size, output_size):
    if isinstance(input_size, tuple):
        input_size = prod(input_size)
    if isinstance(output_size, tuple):
        output_size = prod(output_size)

    weight = he_normal()(key, (input_size//output_size, output_size))
    bias = jnp.zeros(output_size)

    def apply(x, weight, bias):
        x = x.reshape(-1, input_size//output_size, output_size)
        return jnp.einsum('ijk, jk -> ik', x, weight) + bias

    return apply, weight, bias


def conv(key, input_channel, output_channel, kernel_shape, stride, padding="SAME", input_dilation=None, kernel_dilation=None):
    weight = he_normal()(key, (*kernel_shape, input_channel, output_channel))
    bias = jnp.zeros((1,))
    dimensions = ('NHWC', 'HWIO', 'NHWC')

    #print(f"{input_channel=}, {output_channel=}, {kernel_shape=}, {stride=}, {weight.shape=}, {bias.shape=}")

    def apply(x, w, b):
        out = lax.conv_general_dilated(
            x, w, stride, padding,
            input_dilation, kernel_dilation,
            dimensions
        )
        return out + b

    return apply, weight, bias

def local_conv(key, input_channel, output_shape, kernel_shape, stride, padding="VALID", input_dilation=None, kernel_dilation=None):
    fused_input = input_channel * kernel_shape[0] * kernel_shape[1]
    weight = he_normal()(key, (output_shape[0], output_shape[1], fused_input, output_shape[2]))
    bias = jnp.zeros((output_shape[2],)) 
    dimensions = ('NHWC', 'HWIO', 'NHWC')

    #print(f"{input_channel=}, {output_shape=}, {kernel_shape=}, {stride=}, {weight.shape=}, {bias.shape=}")

    def apply(x, w, b):
        out = lax.conv_general_dilated_local(
            x, w, stride, padding, kernel_shape, 
            input_dilation, kernel_dilation,
            dimensions
        )
        return out + b

    return apply, weight, bias

def bilinear_sampler(img, coords):
    H, W = img.shape[1:3]
    x, y = coords[..., 1], coords[..., 0]
    x0, y0 = jnp.floor(x).astype(jnp.int32), jnp.floor(y).astype(jnp.int32)
    x1, y1 = x0 + 1, y0 + 1
    x0, x1 = jnp.clip(x0, 0, W - 1), jnp.clip(x1, 0, W - 1)
    y0, y1 = jnp.clip(y0, 0, H - 1), jnp.clip(y1, 0, H - 1)

    Ia, Ib = img[:, y0, x0, :], img[:, y1, x0, :]
    Ic, Id = img[:, y0, x1, :], img[:, y1, x1, :]
    wa, wb = (x1 - x) * (y1 - y), (x1 - x) * (y - y0)
    wc, wd = (x - x0) * (y1 - y), (x - x0) * (y - y0)
    return Ia * wa[..., None] + Ib * wb[..., None] + Ic * wc[..., None] + Id * wd[..., None]

def flexi(key, input_shape, output_channel, kernel_shape, output_grid):
    k1, k_idx = jr.split(key)
    oh, ow = output_grid
    
    indices = jr.uniform(k_idx, (oh, ow, *kernel_shape, 2))
    indices = indices * jnp.array([input_shape[0], input_shape[1]])

    weight = he_normal()(k1, (*kernel_shape, input_shape[-1], output_channel))
    bias = jnp.zeros((1,))

    def apply(x, w, b, idx):
        patches = bilinear_sampler(x, idx)
        return jnp.einsum('nijklc, klco -> nijo', patches, w) + b

    return apply, weight, bias, indices

def local_flexi(key, input_shape, output_channel, kernel_shape, output_grid):
    k1, k_idx = jr.split(key)
    oh, ow = output_grid
    
    indices = jr.uniform(k_idx, (oh, ow, *kernel_shape, 2))
    indices = indices * jnp.array([input_shape[0], input_shape[1]])

    weight = he_normal()(k1, (oh, ow, *kernel_shape, input_shape[-1], output_channel))
    bias = jnp.zeros((output_channel,))

    def apply(x, w, b, idx):
        patches = bilinear_sampler(x, idx)
        return jnp.einsum('nijklc, ijklco -> nijo', patches, w) + b

    return apply, weight, bias, indices

def squareish_shape(n):
    limit = isqrt(n)
    for h in range(limit, 0, -1):
        if n % h == 0:
            w = n // h
            return (h, w)
    return (1, n)

def conv_output_shape(input_shape, output_channel, kernel_shape, stride, padding="VALID"):
    ih, iw, _ = input_shape
    kh, kw = kernel_shape
    sh, sw = stride
    oc = output_channel

    if padding == "VALID":
        oh = (ih-kh) // sh + 1
        ow = (iw-kw) // sw + 1
        return oh, ow, oc
    elif padding == "SAME":
        oh = (ih + sh - 1) // sh
        ow = (iw + sw - 1) // sw
        return oh, ow, oc
    else:
        raise ValueError

def best_stride_and_channels(input_shape, kernel_shape, target_volume, padding="SAME"):
    candidates = []
    
    max_h = input_shape[0]//2
    max_w = input_shape[1]//2

    for sh, sw in product(range(1, max_h + 1), range(1, max_w + 1)):
        oh, ow, _ = conv_output_shape(input_shape, None, kernel_shape, (sh, sw), padding)
        
        if oh*ow == 0:
            continue
        
        ch = int(ceil(target_volume / oh*ow))
        candidates.append((
            abs(oh*ow*ch - target_volume), 
            sh, sw, (oh, ow, ch)
        ))

    _, sh, sw, output_shape = min(candidates, key=lambda x: (x[0], abs(x[1] - x[2])))
    
    return (sh, sw), output_shape

def maskable_ann_layer(key, nsyns, dends, soma, input_shape):
    key, k1, k2 = jr.split(key, 3)
    sd_f, sd_w, sd_b = linear(k1, input_shape, dends*soma)
    ds_f, ds_w, ds_b = linear(k2, dends*soma, soma)
    output_shape = (soma,)
    return key, LayerParams(sd_w, sd_b, ds_w, ds_b), LayerOps(sd_f, ds_f), output_shape

def small_ann_layer(key, nsyns, dends, soma, input_shape):
    key, k1, k2 = jr.split(key, 3)
    sd_f, sd_w, sd_b = linear(k1, input_shape, dends)
    ds_f, ds_w, ds_b = linear(k2, dends, soma)
    output_shape = (soma,)
    return key, LayerParams(sd_w, sd_b, ds_w, ds_b), LayerOps(sd_f, ds_f), output_shape

def conv_somatic_layer(key, nsyns, dends, soma, input_shape):
    key, k1, k2 = jr.split(key, 3)
    sd_ks = squareish_shape(nsyns)
    stride = (1, 1)
    intermediate_shape = conv_output_shape(input_shape, dends, sd_ks, stride, "SAME")
    sd_f, sd_w, sd_b = conv(k1, input_shape[-1], intermediate_shape[-1], sd_ks, stride)

    ds_ks = squareish_shape(dends)
    output_shape = conv_output_shape(input_shape, soma, ds_ks, stride, "SAME")
    ds_f, ds_w, ds_b = conv(k2, intermediate_shape[-1], output_shape[-1], ds_ks, stride)

    print(f"{input_shape=}, {sd_ks=}, {stride=}, {intermediate_shape=}, {ds_ks=}, {output_shape=}")

    return key, LayerParams(sd_w, sd_b, ds_w, ds_b), LayerOps(sd_f, ds_f), output_shape

def conv_dendritic_layer(key, nsyns, dends, soma, input_shape):
    key, k1, k2 = jr.split(key, 3)
    sd_ks = squareish_shape(nsyns)
    stride, intermediate_shape = best_stride_and_channels(input_shape, sd_ks, dends, "SAME")
    sd_f, sd_w, sd_b = conv(k1, input_shape[-1], intermediate_shape[-1], sd_ks, stride)
    ds_f, ds_w, ds_b = linear(k2, prod(intermediate_shape), soma)
    output_shape = (soma,)

    print(f"{input_shape=}, {sd_ks=}, {stride=}, {intermediate_shape=}, {output_shape=}")

    return key, LayerParams(sd_w, sd_b, ds_w, ds_b), LayerOps(sd_f, ds_f), output_shape

def local_conv_somatic_layer(key, nsyns, dends, soma, input_shape):
    key, k1, k2 = jr.split(key, 3)
    sd_ks = squareish_shape(nsyns)
    stride = (1, 1)
    intermediate_shape = conv_output_shape(input_shape, 1, sd_ks, stride)
    print(intermediate_shape)
    sd_f, sd_w, sd_b = local_conv(k1, input_shape[-1], intermediate_shape, sd_ks, stride)

    ds_ks = squareish_shape(dends)
    stride = ds_ks
    output_shape = conv_output_shape(intermediate_shape, 1, ds_ks, stride)
    if any(x==0 for x in output_shape):
        stride = (1, 1)
        output_shape = conv_output_shape(intermediate_shape, ds_ks, stride)
    ds_f, ds_w, ds_b = local_conv(k2, intermediate_shape[-1], output_shape, ds_ks, stride)

    print(f"{input_shape=}, {sd_ks=}, {stride=}, {intermediate_shape=}, {ds_ks=}, {output_shape=}")

    return key, LayerParams(sd_w, sd_b, ds_w, ds_b), LayerOps(sd_f, ds_f), output_shape

def local_conv_dendritic_layer(key, nsyns, dends, soma, input_shape):
    key, k1, k2 = jr.split(key, 3)
    sd_ks = squareish_shape(nsyns)
    stride, intermediate_shape = best_stride_and_channels(input_shape, sd_ks, dends)
    sd_f, sd_w, sd_b = local_conv(k1, input_shape[-1], intermediate_shape, sd_ks, stride)
    ds_f, ds_w, ds_b = linear(k2, prod(intermediate_shape), soma)
    output_shape = (soma,)

    print(f"{input_shape=}, {sd_ks=}, {stride=}, {intermediate_shape=}, {output_shape=}")

    return key, LayerParams(sd_w, sd_b, ds_w, ds_b), LayerOps(sd_f, ds_f), output_shape

def flexi_somatic_layer(key, nsyns, dends, soma, input_shape):
    key, k1, k2 = jr.split(key, 3)
    sd_ks = squareish_shape(nsyns)
    
    stride = (1, 1)
    int_shape = conv_output_shape(input_shape, dends, sd_ks, stride, "SAME")
    
    sd_f, sd_w, sd_b, sd_idx = flexi(k1, input_shape, int_shape[-1], sd_ks, int_shape[:2])

    ds_ks = squareish_shape(dends)
    output_shape = conv_output_shape(input_shape, soma, ds_ks, stride, "SAME")
    ds_f, ds_w, ds_b = conv(k2, int_shape[-1], output_shape[-1], ds_ks, stride)

    print(f"{input_shape=}, {sd_ks=}, {int_shape=}, {ds_ks=}, {output_shape=}")
    
    return key, LayerParams(sd_w, sd_b, ds_w, ds_b), LayerOps(sd_f, ds_f), output_shape, sd_idx

def flexi_dendritic_layer(key, nsyns, dends, soma, input_shape):
    key, k1, k2 = jr.split(key, 3)
    sd_ks = squareish_shape(nsyns)
    
    stride, int_shape = best_stride_and_channels(input_shape, sd_ks, dends, "SAME")
    
    sd_f, sd_w, sd_b, sd_idx = flexi(k1, input_shape, int_shape[-1], sd_ks, int_shape[:2])
    ds_f, ds_w, ds_b = linear(k2, prod(int_shape), soma)
    output_shape = (soma,)

    print(f"{input_shape=}, {sd_ks=}, {stride=}, {int_shape=}, {output_shape=}")

    return key, LayerParams(sd_w, sd_b, ds_w, ds_b), LayerOps(sd_f, ds_f), output_shape, sd_idx

def local_flexi_somatic_layer(key, nsyns, dends, soma, input_shape):
    key, k1, k2 = jr.split(key, 3)
    sd_ks = squareish_shape(nsyns)
    
    stride = (1, 1)
    int_shape_calc = conv_output_shape(input_shape, 1, sd_ks, stride) 
    int_shape = (*int_shape_calc[:2], dends)

    sd_f, sd_w, sd_b, sd_idx = local_flexi(k1, input_shape, dends, sd_ks, int_shape[:2])

    ds_ks = squareish_shape(dends)
    stride = ds_ks
    output_shape = conv_output_shape(int_shape, 1, ds_ks, stride)
    
    if any(x==0 for x in output_shape):
        stride = (1, 1)
        output_shape = conv_output_shape(int_shape, ds_ks, stride)
        
    ds_f, ds_w, ds_b = local_conv(k2, int_shape[-1], output_shape, ds_ks, stride)

    print(f"{input_shape=}, {sd_ks=}, {int_shape=}, {ds_ks=}, {output_shape=}")

    return key, LayerParams(sd_w, sd_b, ds_w, ds_b), LayerOps(sd_f, ds_f), output_shape, sd_idx

def local_flexi_dendritic_layer(key, nsyns, dends, soma, input_shape):
    key, k1, k2 = jr.split(key, 3)
    sd_ks = squareish_shape(nsyns)
    
    stride, int_shape = best_stride_and_channels(input_shape, sd_ks, dends)
    
    sd_f, sd_w, sd_b, sd_idx = local_flexi(k1, input_shape, int_shape[-1], sd_ks, int_shape[:2])
    ds_f, ds_w, ds_b = linear(k2, prod(int_shape), soma)
    output_shape = (soma,)

    print(f"{input_shape=}, {sd_ks=}, {stride=}, {int_shape=}, {output_shape=}")

    return key, LayerParams(sd_w, sd_b, ds_w, ds_b), LayerOps(sd_f, ds_f), output_shape, sd_idx

# def build_layer(key, config, i, current_shape):
#     key, k1, k2 = jr.split(key, 3)
#     
#     nsyns = config.nsyns[i]
#     dends = config.dends[i]
#     soma = config.soma[i]
# 
#     if config.conventional or config.original: # fully connected
#         # synapse/soma -> dendrite
#         sd_f, sd_w, sd_b = linear(k1, current_shape, dends*soma)
# 
#         # dendrite -> soma
#         ds_f, ds_w, ds_b = linear(k2, dends*soma, soma)
#         
#         current_shape = (soma,)
# 
#     elif config.rfs and config.local: # local rfs
#         kernel_shape = squareish_shape(nsyns)
#         
#         if config.rfs == "dendritic":
#             stride, intermediate_shape = best_stride(current_shape, kernel_shape, dends*soma)
#             intermediate_shape = squareish_shape(intermediate_shape)
#         else:
#             stride = (1, 1)
#             intermediate_shape = conv_output_shape(current_shape, kernel_shape, stride)
# 
#         intermediate_shape = (*intermediate_shape, 1)
# 
#         # synapse/soma -> dendrite
#         sd_f, sd_w, sd_b = conv(k1, current_shape, intermediate_shape, kernel_shape, stride, config.local)
# 
#         # dendrite -> soma
#         if config.rfs == "dendritic": # should be block diagonal*
#             ds_f, ds_w, ds_b = linear(k2, prod(intermediate_shape), soma) # make this block diagonal
#             current_shape = (soma,)
#         else:
#             dend_kernel = squareish_shape(dends)
#             current_shape = conv_output_shape(intermediate_shape, dend_kernel, stride)
#             ds_f, ds_w, ds_b = conv(k2, intermediate_shape, (*current_shape, 1), dend_kernel, stride, config.local)
# 
#     elif config.rfs and not config.local: # non local rfs
#         kernel_shape = squareish_shape(nsyns)
#         
#         intermediate_shape = current_shape[:-1]
#         if config.rfs == "dendritic":
#             stride, _ = best_stride(current_shape, kernel_shape, dends*soma, "SAME")
#             intermediate_shape += (1,)
#         else:
#             stride = (1, 1)
#             intermediate_shape += (dends,)
# 
#         # synapse/soma -> dendrite
#         sd_f, sd_w, sd_b = conv(k1, current_shape, intermediate_shape, kernel_shape, stride, config.local, "SAME", pooling=True)
# 
#         
#         # dendrite -> soma
#         if config.rfs == "dendritic": # should be block diagonal*
#             ds_f, ds_w, ds_b = linear(k2, prod(intermediate_shape[:-1]), soma) # todo make this block diagonal
#             current_shape = (soma,)
#         else:
#             dend_kernel = squareish_shape(dends)
#             current_shape = (*intermediate_shape[:-1], soma)
#             ds_f, ds_w, ds_b = conv(k2, (*intermediate_shape[:-1], 1), current_shape, dend_kernel, stride, config.local, "SAME", pooling=True)
#             current_shape = (*intermediate_shape[:-1], 1)
# 
#     else:
#         raise ValueError
# 
#     return key, LayerParams(sd_w, sd_b, ds_w, ds_b), LayerOps(sd_f, ds_f), current_shape

def get_model(key, config):
    params = []
    ops = []
    masks = []
    indices = []

    key, mask_key = jr.split(key)
    
    if config.original:
        mask_list = get_masks(key, config)
    # elif config.improved:
    #     index_list = get_indices(key, config)

    current_shape = config.input_shape
    if config.original:
        fn = maskable_ann_layer
    elif config.conventional:
        fn = small_ann_layer
    elif not config.local and not config.flexi and config.rfs == "somatic":
        fn = conv_somatic_layer
    elif not config.local and not config.flexi and config.rfs == "dendritic":
        fn = conv_dendritic_layer
    elif config.local and not config.flexi and config.rfs == "somatic":
        fn = local_conv_somatic_layer
    elif config.local and not config.flexi and config.rfs == "dendritic":
        fn = local_conv_dendritic_layer
    elif not config.local and config.flexi and config.rfs == "somatic":
        fn = flexi_somatic_layer
    elif not config.local and config.flexi and config.rfs == "dendritic":
        fn = flexi_dendritic_layer
    elif config.local and config.flexi and config.rfs == "somatic":
        fn = local_flexi_somatic_layer
    elif config.local and config.flexi and config.rfs == "dendritic":
        fn = local_flexi_dendritic_layer

    # TODO flexi

    for i, layer_shape in enumerate(zip(config.nsyns, config.dends, config.soma)):
        key, layer_params, layer_ops, current_shape, *idxs = fn(key, *layer_shape, current_shape)

        ops.append(layer_ops)
        params.append(layer_params)
        if config.original:
            masks.append(LayerMasks(*mask_list[i*2:i*2+2]))
        indices += idxs

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
    ) # + idxs

    print(f"params={total_size}")
    if masks:
        print(f"masked={total_masked}\nactive={total_size-total_masked}")
    #exit()

    def predict(x, params, final_params, indices, dropout_key=None):
        if dropout_key is not None:
            keys = jr.split(dropout_key, len(config.layers)*2)

        # kinda dumb :I would be nice if we preprocessed it to be this way :) 
        x = jnp.array(x, dtype=jnp.float32)
        print(x.shape)

        if indices:
            indices = iter(indices)
            print

        for i, layer in enumerate(params):
            # synapse/soma -> dendrite
            sd_w = layer.sd_w*masks[i].sd if masks else layer.sd_w
            args = (next(indices),) if indices else ()
            x = ops[i].sd(x, sd_w, layer.sd_b, *args)
            if dropout_key is not None:
                keep = jr.bernoulli(keys[i], config.drop_rate, x.shape)
                x = jnp.where(keep,x / config.drop_rate, 0)
            x = leaky_relu(x, 0.1)
            print(x.shape)

            # dendrite -> soma
            ds_w = layer.ds_w*masks[i].ds if masks else layer.ds_w
            x = ops[i].ds(x, ds_w, layer.ds_b)
            if dropout_key is not None:
                keep = jr.bernoulli(keys[i], config.drop_rate, x.shape)
                x = jnp.where(keep,x / config.drop_rate, 0)
            x = leaky_relu(x, 0.1)
            print(x.shape)

        return f_final(x, *final_params)

    return key, predict, params, final_params, indices


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import prod
import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
import optax
import timeit
from jax_peak_memory_monitor import PeakMemoryMonitor

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

    if config.sigma:
        unique_vals += f"noise_{config.sigma}"

    # postfix
    unique_vals += [str(config.trial)]
    return "_".join(unique_vals)

def update_dends_and_soma(config):
    if len(config.nsyns) == 1 and config.layers > 1:
        config.nsyns *= config.layers

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
        'conv_global_rfs', 'conv_local_rfs', 'local_conv_global_rfs', 'local_conv_local_rfs',
        'flexi_patches_global_rfs', 'flexi_patches_local_rfs', 
        'local_flexi_patches_global_rfs', 'local_flexi_patches_local_rfs', 
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

    if config.model_id < 12:
        config.original = True
    
    match config.model_id:
        case 0:
            pass
        case 1:
            config.rfs = "somatic"
        case 2:
            config.rfs = "dendritic"
        case 3:
            config.conventional = True
            config.original = False # don't mask.
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
            config.rfs = "somatic"
        case 13:
            config.rfs = "dendritic"
        case 14:
            config.local = True
            config.rfs = "somatic"
        case 15:
            config.local = True
            config.rfs = "dendritic"
        case 16:
            config.flexi = True
            config.rfs = "somatic"
        case 17:
            config.flexi = True
            config.rfs = "dendritic"
        case 18:
            config.local = True
            config.flexi = True
            config.rfs = "somatic"
        case 19:
            config.local = True
            config.flexi = True
            config.rfs = "dendritic"
        case _:
            raise ValueError
    
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

    if dataset == "fmnist":
        dataset = "fashion_mnist"

    tf_split = [
        f"train[:{split:.0%}]", # train
        f"train[{split:.0%}:]", # validation
        "test", # test
    ]

    loader, info = tfds.load(dataset, split=tf_split, batch_size=batch_size, data_dir=data_dir, as_supervised=True, with_info=True)
    loader = tfds.as_numpy(loader)
    loader = dict(zip(["train", "val", "test"], loader))
    
    return loader, info

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

    #t = jnp.arange(100*32*32*3).reshape(100, 32, 32, 3)
    #_ = model(t/t.max(), *params)
    #return None, None
    
    model = jit(model)

    #exit()

    opt_state = optimizer.init(params)

    # TODO add more than just loss and accuracy.
    res = {
        "train": {
            "loss": [],
            "acc": []
        }, "val": {
            "loss": [],
            "acc": []
        }, "test": {
            "loss": [],
            "acc": [],
            "time": [],
            "memory": []
        }
    }

    for epoch in range(epochs):
        local_loss = None
        local_acc = None
        for i, batch in (t:=tqdm(enumerate(dataloader["train"]), total=len(dataloader["train"]))):
            loss, outputs, params, opt_state = update(params, batch, opt_state)
            acc = sparse_categorical_accuracy_with_integer_labels(outputs, batch[1])

            local_loss = inc_mean(local_loss, loss, i+1)
            local_acc = inc_mean(local_acc, acc, i+1)
            t.set_description(f"{epoch=}, loss={local_loss:,.4f}, acc={local_acc:.2%}")
        
        res["train"]["loss"].append(local_loss)
        res["train"]["acc"].append(local_acc)

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

            res["val"]["loss"].append(local_loss)
            res["val"]["acc"].append(local_acc)

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
    
    res["test"]["loss"].append(local_loss)
    res["test"]["acc"].append(local_acc)

    # n, r = 10, 10
    # times = []
    # with PeakMemoryMonitor(0) as m:
    #     times += timeit.repeat(lambda: [
    #                 model(inputs, *params)
    #                 for inputs, outputs in dataloader["test"]
    #             ], number=n, repeat=r)
    #     print(times)
    # #print(f"Speed after compilation: batches={n*len(dataloader)/times.mean():,.2f}±{len(dataloader)*times.std():.2f}it/s")

    # print(times[0])
    # print(m.peak)
    # res["test"]["times"] = times
    # res["test"]["memory"] = m.peak

    return params, res


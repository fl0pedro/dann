from keras import backend as K
from keras import layers, initializers
from keras.activations import relu, softmax
from keras.saving import register_keras_serializable
import keras
import numpy as np
from typing import Iterable

class LocallyConnected2D(layers.Layer):
    def __init__(self, in_channels, out_channels, input_shape,
                 kernel_size, stride, bias=True, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, Iterable) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, Iterable) else (stride, stride)
        self.bias = bias
        self.input_shape = tuple(input_shape) + (in_channels,)

        ih, iw = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride

        oh = (ih + 2 * (kh // 2) - kh) // sh + 1
        ow = (iw + 2 * (kw // 2) - kw) // sw + 1

        self.output_shape = (oh, ow, out_channels)
        self.weight_shape = (out_channels, in_channels, oh, ow, kh * kw)

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=self.weight_shape,
            initializer=initializers.RandomNormal(stddev=0.05),
            trainable=True,
            name='weight'
        )
        if self.bias:
            self.bias = self.add_weight(
                shape=self.output_shape,
                initializer='zeros',
                trainable=True,
                name='bias'
            )
        else:
            self.bias = None

    def call(self, x):
        backend = K.backend()
        if backend == "tensorflow":
            return self.tensorflow_locally_connected_2d(x)
        elif backend == "torch":
            return self.torch_locally_connected_2d(x)
        elif backend == "jax":
            return self.jax_locally_connected_2d(x)
        else:
            raise NotImplementedError(f"Unsupported backend: {backend}")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "input_shape": self.input_shape,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "bias": self.bias is not None,
        })
        return config

    def tensorflow_locally_connected_2d(self, x):
        import tensorflow as tf

        B = tf.shape(x)[0]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow, _ = self.output_shape

        ph = kh // 2
        pw = kw // 2
        x = tf.pad(x, [[0, 0], [ph, ph], [pw, pw], [0, 0]])

        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, kh, kw, 1],
            strides=[1, sh, sw, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, [B, oh, ow, self.in_channels, kh * kw])
        patches = tf.transpose(patches, [0, 3, 1, 2, 4])  # [B, C, oh, ow, kh*kw]
        patches = tf.expand_dims(patches, axis=1)  # [B, 1, C, oh, ow, kh*kw]

        output = tf.reduce_sum(patches * self.weight[tf.newaxis, ...], axis=[2, 5])  # [B, out_ch, oh, ow]
        output = tf.transpose(output, [0, 2, 3, 1])  # [B, oh, ow, out_ch]
        if self.bias is not None:
            output += self.bias[tf.newaxis, ...]
        return output

    def torch_locally_connected_2d(self, x):
        import torch

        B, C, *_ = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow, _ = self.output_shape

        ph = kh // 2
        pw = kw // 2
        x = torch.nn.functional.pad(x, (pw, pw, ph, ph))

        # Unfold returns shape: [B, C*kh*kw, oh*ow]
        patches = torch.nn.functional.unfold(x, (kh, kw), stride=(sh, sw))
        patches = patches.view(B, C, kh * kw, oh, ow)  # [B, C, kh*kw, oh, ow]
        patches = patches.permute(0, 1, 3, 4, 2)  # [B, C, oh, ow, kh*kw]
        patches = patches.unsqueeze(1)  # [B, 1, C, oh, ow, kh*kw]

        out = (patches * self.weight).sum([2, 5])  # [B, out_ch, oh, ow]
        out = out.permute(0, 2, 3, 1)  # [B, oh, ow, out_ch]
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out

    def jax_locally_connected_2d(self, x):
        import jax
        import jax.numpy as jnp

        kh, kw = self.kernel_size
        sh, sw = self.stride

        ph = kh // 2
        pw = kw // 2

        if x.ndim == 3:
            x = x[jnp.newaxis, ...]  # Add batch dim if missing

        x = jnp.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw), (0, 0)))  # NHWC
        x = jnp.transpose(x, (0, 3, 1, 2))  # NCHW

        def extract_patches(x_img):  # [C, H, W]
            patches = []
            for i in range(0, x_img.shape[1] - kh + 1, sh):
                row = []
                for j in range(0, x_img.shape[2] - kw + 1, sw):
                    patch = x_img[:, i:i+kh, j:j+kw].reshape(self.in_channels, -1)
                    row.append(patch)
                patches.append(jnp.stack(row, axis=1))
            return jnp.stack(patches, axis=1)  # [C, oh, ow, kh*kw]

        patches = jax.vmap(extract_patches)(x)  # [B, C, oh, ow, kh*kw]
        patches = patches[:, None, :, :, :, :]  # [B, 1, C, oh, ow, kh*kw]

        out = jnp.sum(patches * self.weight, axis=(2, 5))  # [B, out_ch, oh, ow]
        out = jnp.transpose(out, (0, 2, 3, 1))  # [B, oh, ow, out_ch]
        if self.bias is not None:
            out += self.bias[jax.newaxis, ...]
        return out

    
@register_keras_serializable()
class MultiLayerLocallyConnected2D(keras.Model):
    def __init__(self, input_shape, layer_depth, output_size,
                 kernels, strides, bias=True, **kwargs):
        super().__init__(**kwargs)

        self.input_shape = input_shape
        self.layer_depth = layer_depth
        self.output_size = output_size
        self.kernels = kernels
        self.strides = strides
        self.bias = bias
        
        if isinstance(kernels, tuple):
            kernels = [kernels] * (len(layer_depth) - 1)
        if isinstance(strides, tuple):
            strides = [strides] * (len(layer_depth) - 1)

        shapes = [input_shape]
        for i in range(len(layer_depth) - 1):
            ih, iw = shapes[-1]
            kh, kw = kernels[i]
            sh, sw = strides[i]
            oh = (ih + 2 * (kh // 2) - kh) // sh + 1
            ow = (iw + 2 * (kw // 2) - kw) // sw + 1
            shapes.append((oh, ow))

        self.lcs = []
        for i in range(len(layer_depth) - 1):
            self.lcs.append(
                LocallyConnected2D(
                    in_channels=layer_depth[i],
                    out_channels=layer_depth[i+1],
                    input_shape=shapes[i],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    bias=bias,
                    name=f"lc_{i}"
                )
            )

        final_h, final_w = shapes[-1]
        final_channels = layer_depth[-1]
        self.flatten = keras.layers.Flatten()
        self.linear = keras.layers.Dense(output_size)
    
    def call(self, x):
        # x expected in [B, H, W, C] format
        for lc in self.lcs:
            x = relu(lc(x))
        x = self.flatten(x)
        x = self.linear(x)
        return softmax(x, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape,
            "layer_depth": self.layer_depth,
            "output_size": self.output_size,
            "kernels": self.kernels,
            "strides": self.strides,
            "bias": self.bias,
        })
        return config

if __name__ == "__main__":
    model = MultiLayerLocallyConnected2D(
        input_shape=(32, 32),
        layer_depth=[3, 16, 32],
        output_size=10,
        kernels=(3, 3),
        strides=(1, 1),
        bias=True
    )


    dummy_input = np.random.normal(size=(8, 32, 32, 3))  # [batch, H, W, C]
    output = model(dummy_input)

    print(model.summary())
    print(output.shape)  # (8, 10)

import numpy as np
import time
from tinygrad import Tensor, nn, TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.datasets import mnist
from tinygrad.helpers import prod, colored, trange, argsort


def unfold(x:Tensor, dim:int, size:int, step:int) -> Tensor:
  """
  Unfolds the tensor along dimension `dim` into overlapping windows.

  Each window has length `size` and begins every `step` elements of `x`.
  Returns the input tensor with dimension `dim` replaced by dims `(n_windows, size)`
  where `n_windows = (x.shape[dim] - size) // step + 1`.

  ```python exec="true" source="above" session="tensor" result="python"
  unfolded = Tensor.arange(8).unfold(0,2,2)
  print("\\n".join([repr(x.numpy()) for x in unfolded]))
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  unfolded = Tensor.arange(27).reshape(3,3,3).unfold(-1,2,3)
  print("\\n".join([repr(x.numpy()) for x in unfolded]))
  ```
  """
  if size < 0: raise RuntimeError(f'size must be >= 0 but got {size=}')
  if step <= 0: raise RuntimeError(f'step must be > 0 but got {step=}')
  if size > x.shape[dim]: raise RuntimeError(f'maximum size for tensor at dimension {dim} is {x.shape[dim]} but size is {size}')
  dim = x._resolve_dim(dim)
  perm_to_last = tuple(i for i in range(x.ndim) if i != dim) + (dim,)
  return x.permute(perm_to_last)._pool((size,), step).permute(argsort(perm_to_last) + (x.ndim,))


class LocallyConnected2d:
    def __init__(self, in_ch, out_ch, kh, kw, oh, ow, sh=1, sw=1, bias = True):
        self.kh, self.kw, self.sh, self.sw = kh, kw, sh, sw
        self.oh, self.ow = oh, ow
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.weight = Tensor.randn(in_ch * kh * kw, out_ch, oh, ow) * (
            1.0 / np.sqrt(in_ch * kh * kw)
        )
        if bias:
            self.bias = Tensor.zeros(out_ch, oh, ow)  # Bias per output channel and location
            self.conv = lambda patches: Tensor.einsum(
                    "bkij,koij->boij", patches, self.weight
                ) + self.bias.unsqueeze(0)
        else:
            self.conv = lambda patches: Tensor.einsum(
                    "bkij,koij->boij", patches, self.weight
                )

    def __call__(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.reshape(1, *x.shape)
        patches = unfold(x, 2, self.kw, self.sw)
        patches = unfold(patches, 3, self.kh, self.sh)
        # patches = x.unfold(2, self.kh, self.sh).unfold(3, self.kw, self.sw)
        out = self.conv(patches) 
        return out


class MultiLayerLocallyConnected2D:
    def __init__(
        self,
        input_shape: tuple[int, int],
        layer_depth: list[int],
        output_size: int,
        kernels: tuple[int, int] | list[tuple[int, int]],
        strides: tuple[int, int] | list[tuple[int, int]],
        bias: bool = True,
    ):
        assert len(layer_depth) >= 2
        self.lcs = []
        H, W = input_shape

        if isinstance(kernels[0], int):
            kernels = [kernels] * (len(layer_depth) - 1)
        if isinstance(strides[0], int):
            strides = [strides] * (len(layer_depth) - 1)

        for in_ch, out_ch, k, s in zip(
            layer_depth[:-1], layer_depth[1:], kernels, strides
        ):
            kh, kw = k
            sh, sw = s
            oh = (H - kh) // sh + 1
            ow = (W - kw) // sw + 1
            self.lcs.append(LocallyConnected2d(in_ch, out_ch, kh, kw, oh, ow, sh, sw, bias))
            H, W = oh, ow

        self.output_H, self.output_W = H, W
        self.final_ch = layer_depth[-1]
        self.weight = Tensor.randn(self.final_ch * H * W, output_size) * (
            1.0 / np.sqrt(self.final_ch * H * W)
        )

        if bias:
            self.bias = Tensor.zeros(output_size)
            self.linear = lambda x: x.dot(self.weight) + self.bias
        else:
            self.linear = lambda x: x.dot(self.weight)

    def __call__(self, x: Tensor) -> Tensor:
        for lc in self.lcs:
            x = lc(x).relu()
        x = x.flatten(1)
        return self.linear(x)

class TinyDANN:
    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_layers: int,
        dends: list[int],
        soma: list[int],
        output_size: int,
        fname_model: str,
        dropout: bool,
        rate: float,
    ):
        self.ls = []
        for i, (d, s) in enumerate(zip(dends, soma)):
            in_shape = prod(input_shape) if i == 0 else s
            self.ls.append(nn.Linear(in_shape, d * s))
            self.ls.append(nn.Linear(d * s, s))
        self.out = nn.Linear(s, output_size)
        self.rate = rate if dropout else 0.

    def __call__(self, x: Tensor):
        x = x.reshape(x.shape[0], -1)
        for layer in self.ls:
            x = layer(x).relu().dropout(self.rate)
        return self.out(x).softmax()

def train_loop(
    model,
    masks,
    batch_size,
    num_epochs,
    x_train,
    x_val,
    x_test,
    y_train,
    y_val,
    y_test,
    learning_rate,
    shuffle,
):
    if masks is not None:
        for i in range(len(model.ls)):
            model.ls[i].weight *= masks[i*2].T
            model.ls[i].bias *= masks[i*2+1]
        model.out.weight *= masks[-2].T
        model.out.bias *= masks[-1]

    opt = nn.optim.Adam(nn.state.get_parameters(model), learning_rate)

    @TinyJit
    @Tensor.train()
    def train_step(xb, yb) -> Tensor:
        opt.zero_grad()
        loss = model(xb).sparse_categorical_crossentropy(yb).backward()
        opt.step()
        return loss

    def get_loss(x: Tensor, y: Tensor) -> Tensor:
        return model(x).sparse_categorical_crossentropy(y).mean()

    def get_acc(x: Tensor, y: Tensor) -> Tensor:
        return (model(x).argmax(axis=x.ndim-3) == y).mean()

    n_train = x_train.shape[0]
    t = trange(7000)

    # Preallocate a tensor for indices (avoid realloc every loop)
    idx_buffer = Tensor.empty(batch_size)

    t0 = time.time()
    for step in t:
        # Sample random indices on CPU once per step
        idx_buffer = Tensor.randint(batch_size, high=n_train)
        x_batch = x_train[idx_buffer]
        y_batch = y_train[idx_buffer]

        loss = train_step(x_batch, y_batch)

        # Cache item once for performance
        loss_item = loss.item()
        acc = get_acc(x_batch, y_batch)
        acc_item = acc.item()

        t.set_description(f"loss {loss_item:.4f}, acc {acc_item:.2%}")

        if step % 100 == 0:
            val_loss = get_loss(x_val, y_val).item()
            val_acc = get_acc(x_val, y_val).item()
            t.write(f"{step=} - val loss {val_loss:.4f}, val acc {val_acc:.2%}")
        
    test_loss = get_loss(x_test, y_test).item()
    test_acc = get_acc(x_test, y_test).item()
    print(f"test loss {test_loss:.4f}, test acc {test_acc:.2%}")
    print(f"time: {time.time() - t0:.2f}s")

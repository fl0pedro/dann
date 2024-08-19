# Chavlis_Poirazi_2024
Dendritic Artificial Neural Networks (dANNs) with Receptive Fields (RFs)

These codes replicate: Chavlis, S., & Poirazi, P (2024). Dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning. [arXiv:2404.03708v1](https://arxiv.org/abs/2404.03708v1)

To replicate the Figures of the manuscript, you need to install the Anaconda environment (see [here](https://www.anaconda.com/download/))

1. Download the executable file and install it. Then, create a new environment from a terminal upon activation of anaconda (i.e., ```conda activate```)
2. ```conda env create -f environment.yml```
3. ```conda activate dann```

You can run the files `.py` with `python figure_2.py`, for example, or train the model using the `sh files`.


**GPU support**
You need to install NVIDIA driver, CUDA 12.1 and then install tensorflow, pytorch and jax with cuda compatibility.

You can find your CUDA version with the command:
```nvcc --version```

### Tensorflow (https://www.tensorflow.org/install)
```python3 -m pip install tensorflow[and-cuda]```

and verify the installation:
```python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"```

### Pytorch (https://pytorch.org/get-started/locally/)
```pip3 install torch torchvision torchaudio```

and verify the installation
```python3 -c "import torch; print(torch.cuda.is_available())"```

### Jax (https://jax.readthedocs.io/en/latest/installation.html)
```pip install -U "jax[cuda12]"```


**CUML installation** 
Assuming CUDA 12.1 is installed and everything run under Linux OS.

```pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12```

# Extract the data

`python unzip_data.py`


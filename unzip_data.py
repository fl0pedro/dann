#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 00:21:24 2024

@author: spiros
"""

import zipfile
with zipfile.ZipFile('DATA.zip', 'r') as zip_ref:
    zip_ref.extractall('')

with zipfile.ZipFile('DATASETS/EMNIST.zip', 'r') as zip_ref:
    zip_ref.extractall('DATASETS/EMNIST/')

with zipfile.ZipFile('DATASETS/KMNIST.zip', 'r') as zip_ref:
    zip_ref.extractall('DATASETS/KMNIST')
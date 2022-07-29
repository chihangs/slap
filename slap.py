#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 04:13:17 2022

@author: suenchihang

#not yet generalized, now only works if x is 2D array
#now slap occurs to whole list of arrays, but need to slap individual array inside x instead
"""

import numpy as np

def slap(x):
    variants={}
    variants['no_flip'] = [np.rot90(x, k=i, axes=(-2, -1)) for i in range(4)]
    y = np.flip(x, axis=-1)  #flip horizontally
    variants['flip'] = [np.rot90(y, k=i, axes=(-2, -1)) for i in range(4)]
    temp = variants['no_flip'][0].tolist()
    temp_flip = 'no_flip'
    temp_i = 0
    for j in ['no_flip', 'flip']:
        for i in range(4):
            candidate = variants[j][i].tolist()
            if candidate > temp:
                temp = candidate
                temp_flip = j
                temp_i = i
    return variants[temp_flip][temp_i], temp_flip, temp_i
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 04:13:17 2022

@author: suenchihang

# refactoring slap and add "unslap"
#not yet generalized, now only works if x is 2D array
#now slap occurs to whole list of arrays, but need to slap individual array inside x instead
"""

import numpy as np
import copy

def slap(x):
    variants={}
    variants['no_flip'] = [np.rot90(x, k=i, axes=(-2, -1)) for i in range(4)]
    y = np.flip(x, axis=-1)  #flip horizontally
    variants['flip'] = [np.rot90(y, k=i, axes=(-2, -1)) for i in range(4)]
    index = max(range(8), key=lambda i:variants['flip' if i//4 else 'no_flip'][i%4].tolist()) 
    temp_flip = 'flip' if index//4 else 'no_flip'
    temp_i = index%4
    return variants[temp_flip][temp_i], temp_flip, temp_i    #the largest variant

def unslap(x, temp_flip, temp_i):
    y = np.rot90(x, k= -temp_i, axes=(-2, -1))    #negative temp_i for reverse
    return y if temp_flip == 'no_flip' else np.flip(y, axis=-1)


def slap_opening(board):    #for first move only
    current_state = np.zeros((board.height, board.width))
    opening = []
    for move in list(range(board.width * board.height)):
        new_state = copy.deepcopy(current_state)
        new_state[move//board.width, move%board.width] = 1
        slap_state, temp_flip, temp_i = slap(new_state)
        if temp_flip=='no_flip' and temp_i == 0:
            opening.append(move)
    return opening   #list of moves in integers 0, 1, 2...
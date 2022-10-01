#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 04:13:17 2022

@author: suenchihang

# slap6: add scale_pos, stone_pos, and use them to refactor cc_state, cc_pos
# slap5: add cc_pos
# slap4: add slap_variants, bbox, centre, cc, cc_state
#     exploratory: min max using ReLU, add slap_pixel, transform_pixel, slap_pixel_list
# slap3: add slap_opening
# refactoring slap and add "unslap"
#not yet generalized, now only works for 2D problems with colour channels
#now slap occurs to whole list of arrays, not individual array inside x instead
"""

import numpy as np
import copy

def slap_variants(x):   #return dict of variants
    variants={}
    variants['no_flip'] = [np.rot90(x, k=i, axes=(-2, -1)) for i in range(4)]
    y = np.flip(x, axis=-1)  #flip horizontally
    variants['flip'] = [np.rot90(y, k=i, axes=(-2, -1)) for i in range(4)]
    return variants

def slap(x):
    variants = slap_variants(x)
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


def bbox(img): #give bounding box min & max
    if np.any(img):  #if non-zeros
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
    else:
        rmin, rmax = (0, img.shape[0]-1)
        cmin, cmax = (0, img.shape[1]-1)
    return (rmin, cmin), (rmax, cmax)

def bbox_colour(img):  #colour version of bbox
    (rmin, cmin), (rmax, cmax) = bbox(img[0])
    for i in range(1, len(img)):
        b_min, b_max = bbox(img[i])
        if rmin > b_min[0]: rmin = b_min[0]
        if cmin > b_min[1]: cmin = b_min[1]
        if rmax < b_max[0]: rmax = b_max[0]
        if cmax < b_max[1]: cmax = b_max[1]
    return (rmin, cmin), (rmax, cmax)

def centre(img, box_min, box_max):   #shift to centre, biased towards top left if can't be exact centre
    r_shift = (img.shape[-2]-1-box_max[0]-box_min[0])//2
    c_shift = (img.shape[-1]-1-box_max[1]-box_min[1])//2
    return np.roll(img, (r_shift, c_shift), axis=(-2, -1))

def cc(img):  #crop and centre
    if img.ndim>2:
        b_min, b_max = bbox_colour(img)
    else:
        b_min, b_max = bbox(img)
    return centre(img, b_min, b_max)

def scale_pos(state):
    """" state: current_state of Gomoku board
        return: scaled position index"""
    height, width = (state.shape[-2], state.shape[-1])
    box_min, box_max = bbox(state[0]-state[1])  #bounding box min & max
    vertical_pos = [[(i*2+1-height)/(height-1)]*height for i in range(height)]  #scaled index
    horizontal_pos = [[(i*2+1-width)/(width-1) for i in range(width)] for i in range(width)]
    return vertical_pos, horizontal_pos


def cc_state(state):
    """" state: current_state of Gomoku board
        return: crop & centre info with scaled position index, same shape as state; 
    if can't be exact centre, slightly baised towards top left"""
    vertical_pos, horizontal_pos = scale_pos(state)
    cc_info = centre(np.stack((state[0],state[1],vertical_pos, horizontal_pos)), box_min, box_max)   #bbox centred
    return cc_info

def stone_pos(state):
    """" state: current_state of Gomoku board
        return: scaled position index, same shape as state; 
        all 4 planes are position indices of only placed stones"""
    vertical_pos, horizontal_pos = scale_pos(state)
    vert0, hori0 = (np.multiply(state[0], vertical_pos), np.multiply(state[0], horizontal_pos))
    vert1, hori1 = (np.multiply(state[1], vertical_pos), np.multiply(state[1], horizontal_pos))
    pos = np.stack((vert0, hori0, vert1, hori1))
    return pos

def cc_pos(state):
    """" state: current_state of Gomoku board
        return: crop & centre info with scaled position index, same shape as state; 
        but all 4 planes are position indices (after cc) of only placed stones
    if can't be exact centre, slightly baised towards top left"""
    pos = stone_pos(state)
    cc_info = centre(pos, box_min, box_max)   #bbox centred
    return cc_info


def relu(x):
    return max(x, 0)

def max_relu(a, b):    #implement max by using relu
    return relu(a-b)+b

def min_relu(a, b):    #implement min by using relu
    return a-relu(a-b) 

def slap_pixel(p, side=8, transform = True):
    #return slap position of a pixel, where p is original position
    #Assume pixel values are non-negative, so earlier position is preferred
    r, c = p
    a = min(r, side-1-r)
    b = min(c, side-1-c)
    r_ = min(a, b)
    c_ = max(a, b)
    #check if transformation needed: 1 yes 0 no, -1 yes and no
    if transform:
        v = int(a<r) if not (r == side-1-r) else -1  #vertical reflection
        h = int(b<c) if not (c == side-1-c) else -1  #horizontal reflection
        t = int(r_<a) if not (r_ == a) else -1   #transpose
        transform = (v, h, t)
        return (r_, c_), transform
    else:
        return (r_, c_)

def transform_pixel(p, transform):
    #carry out transformation in above format, return result position of transformation
    r, c = p
    if transform[0]:
        r = min(r, side-1-r)
    if transform[1]:
        c = min(c, side-1-c)
    if transform[2]:
        return c, r
    else:
        return r, c

def slap_pixel_list(pixel_positions, side=8, pixel_values=None):
    """pixel_positions: list of tuples
       pixel_values: list of pixel values of the same length
    return list of slap positions of each pixel
    not yet completed for tricky cases as this function is not needed in experiment """
    positions_trans = [slap_pixel(p, side, transform=True) for p in pixel_positions]
    index = min(range(len(positions_trans)), key=lambda i:positions_trans[i][0])
    pos_tran = positions_trans.pop(index)
    index2 = min(range(len(positions_trans)), key=lambda i:positions_trans[i][0])
    pos_tran2 = positions_trans.pop(index2)
    if pos_tran[0] < pos_tran2[0]:  #i.e. case for no double winner
        if pixel_values is None:
            if -1 not in pos_tran[1]:   #straight forward if no alternative transformation
                slap_pos = [transform_pixel(p, positions_trans[index][1]) for p in pixel_positions]
            else:
                options = []
                for idx, x in enumerate(pos_tran[1]):
                    if x ==-1:
                        options.append([1, 0])
                    else:
                        options.append([x])
                pass #need to consider 2nd lowest, 3rd and so on. actually also need to check if there are two minimums and see their corresponding 2nd, 3rd etc.
    return slap_pos



        
        
        
        
    
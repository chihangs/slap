#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 04:49:36 2022
generate synthetics states for detecting only immediate win or random
return list of (states, pi, rewards)

@author: suenchihang
"""

import random
import numpy as np
import torch
from game_array2trial import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alpha0_reuse import MCTSPlayer
from policy6b import PolicyValueNet
import time
from slap3 import slap, slap_opening
import copy



#horizontal win
def horizontal_win():
    states, probs = [], []
    for i in range(8):
        for j in range(4):
            availables = list(range(64))
            state = np.zeros((4, 8,8))
            state[0, i, j:j+5] = 1
            state[3] = state[3] + 1
            for k in range(5):
                availables.remove(i*8+j+k)
            for add in range(5):    #reserve this as last stone to add to win
                s = copy.deepcopy(state)
                s[0, i, j+add] = 0 
                pi = np.zeros((8,8))  
                pi[i, j+add] = 1
                if add == 0 and j+5 <= 7:   #check winning at another end
                    pi[i, j+5] = 1
                    move_2 = i*8 + j+5
                elif add==4 and j-1 >= 0:
                    pi[i, j-1] = 1
                    move_2 = i*8 + j-1
                pi = pi.flatten()
                opp_move = np.array(random.sample(availables, 4))  #convert to array for operation below
                if move_2 in opp_move:
                    pi[move_2]=0
                pi /= pi.sum()
                s[1][opp_move//8, opp_move%8] = 1
                s[2][opp_move[-1]//8, opp_move[-1]%8] = 1
                states.append(s)
                probs.append(pi)
    rewards = [1]*8*4*5
    return states, probs, rewards

# diagonal win
def diagonal_win():
    states, probs = [], []
    for i in range(4):
        for j in range(4):
            availables = list(range(64))
            state = np.zeros((4, 8,8))
            state[3] = state[3] + 1
            for k in range(5):
                availables.remove(i*8+j+k*9)
                state[0][i+k, j+k] = 1
            for add in range(5):    #reserve this as last stone to add to win
                s = copy.deepcopy(state)
                s[0, i+add, j+add] = 0 
                pi = np.zeros((8,8))  
                pi[i+add, j+add] = 1
                if add == 0 and j+5 <= 7 and i+5<=7:   #check winning at another end
                    pi[i+5, j+5] = 1
                    move_2 = (i+5)*8 + j+5 
                elif add==4 and j-1 >= 0 and i-1>=0:
                    pi[i-1, j-1] = 1
                    move_2 = (i-1)*8 + j-1
                pi = pi.flatten()
                opp_move = np.array(random.sample(availables, 4))  #convert to array for operation below
                if move_2 in opp_move:
                    pi[move_2] = 0
                pi /= pi.sum()
                s[1][opp_move//8, opp_move%8] = 1
                s[2][opp_move[-1]//8, opp_move[-1]%8] = 1
                states.append(s)
                probs.append(pi)
    rewards = [1]*4*4*5
    return states, probs, rewards
                    
                    
#random labels
def random_state(num_random):
    random_batches = []
    board_dummy = Board(width=8, height=8, n_in_row=5)
    game_dummy = Game(board_dummy)
    board_dummy.init_board(start_player=0)
    for i in range(num_random):
        moves = np.array(random.sample(list(range(64)),8))
        availables = list(set(list(range(64))) - set(moves)) 
        state = np.zeros((4, 8,8))
        state[0][moves[:4]//8, moves[:4]%8] = 1
        state[1][moves[-4:]//8, moves[-4:]%8] = 1
        state[2][moves[-1]//8, moves[-1]%8] = 1
        state[3] = state[3] + 1
        win_moves = board_dummy.winning_move(state[0], availables)   
        if len(win_moves)==0:
            pi = np.random.uniform(size=64)   
            pi[moves] = 0
            pi /= pi.sum()
            reward = 0
        else:
            pi = np.zeros(64)
            pi[np.array(win_moves)]=1/len(win_moves)
            reward = 1
        random_batches.append((state, pi, reward))
    return random_batches


def synthetic_states(num_random, win_set=1):
    """  generate list of (states, pi, rewards) for detecting only immediate win or random"""
    random_batches = random_state(num_random)
    win_states, win_pi, win_rewards = [], [], []
    for i in range(win_set):
        states, pi, rewards = horizontal_win()
        v_states = list(np.rot90(states, k=1, axes=(-2, -1)))
        v_pi = []
        for p in pi:
            v_pi.append(np.rot90(p.reshape((8,8)), k=1, axes=(-2, -1)).flatten())
        v_rewards = rewards
        h_states, h_pi, h_rewards = horizontal_win()
        states, pi, rewards = diagonal_win()
        u_states = list(np.rot90(states, k=1, axes=(-2, -1)))
        u_pi = []
        for p in pi:
            u_pi.append(np.rot90(p.reshape((8,8)), k=1, axes=(-2, -1)).flatten())
        u_rewards = rewards
        d_states, d_pi, d_rewards = diagonal_win()
        # contatenate lists
        win_states = win_states + v_states + h_states + u_states + d_states
        win_pi = win_pi + v_pi + h_pi + u_pi + d_pi
        win_rewards = win_rewards + v_rewards + h_rewards + u_rewards + d_rewards
    
    return random_batches + list(zip(win_states, win_pi, win_rewards))
        
    



            
        
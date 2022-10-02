#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 02:42:35 2022

@author: suenchihang

AI plays against baseline AI for evaluation
AIvsAI_flexible: same code as AIvsAI except config & flexible to change baseline model,
                 add data fields to record, print time & win ratio, add is_shown option
"""

from game_array import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alpha0_reuse import MCTSPlayer
from slap6b import cc_state
import time
from datetime import date, datetime
import numpy as np
import torch
from collections import defaultdict
import csv
import subprocess
import os
import sys

import torch.multiprocessing as mp
try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass

try:
    from jarviscloud import jarviscloud 
    is_jarviscloud = True
except ModuleNotFoundError:
    is_jarviscloud = False

try:
    from autoclip.torch import QuantileClip 
except ModuleNotFoundError:
    if not is_jarviscloud:
        os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"
        os.environ['http_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "autoclip"])

from policy10a import PolicyValueNet   # import after autoclip is installed

def randomize(seed_rng=123, seed_np=123, seed_torch=123):
    os.environ['PYTHONHASHSEED'] = '0'
    rng = np.random.default_rng(seed_rng)
    np.random.seed(seed_np)
    torch.manual_seed(seed_torch)

def AI_compete(n_games, model_file, use_slap, cc_fn, dropout, baseline, is_shown=0):
    # common config
    n = 5   #n in a row to win
    width, height = 8, 8
    num_ResBlock=0
    alpha=0.3 #not affect game as noise is only applied in self-play
    L2=1e-4   #not affect game as it only affects optimizer in self-play
    # set players
    in_channel_baseline = 8 if baseline['cc_fn'] else 4 
    in_channel = 8 if cc_fn else 4    
    policy_baseline = PolicyValueNet(width, height, baseline['model_file'], baseline['use_slap'], num_ResBlock, L2, None, alpha, 'Adam', baseline['dropout'], 0, in_channel_baseline, baseline['cc_fn'])
    policy_tested = PolicyValueNet(width, height, model_file, use_slap, num_ResBlock, L2, None, alpha, 'Adam', dropout, 0, in_channel, cc_fn)
    AI_baseline = MCTSPlayer(policy_baseline.policy_value_fn, c_puct=5, n_playout=400)
    AI_tested = MCTSPlayer(policy_tested.policy_value_fn, c_puct=5, n_playout=400)
    # play against baseline model
    win_cnt = defaultdict(int)
    for i in range(n_games):
        new_board = Board(width=width, height=height, n_in_row=n)
        new_game = Game(new_board)
        winner = new_game.start_play(AI_tested, AI_baseline, start_player=i % 2, is_shown=is_shown)
        win_cnt[winner] += 1
    win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
    print("win: {}, lose: {}, tie:{}".format(win_cnt[1], win_cnt[2], win_cnt[-1]))
    return win_ratio


def run(cpu_n, model_files, use_slaps, cc_fns, dropouts, file_name, baseline, is_shown=0):

    with open('./data/'+file_name+'.csv', 'w', encoding='UTF8', newline='') as f:
        csv.writer(f).writerow(['win_ratio', 'duration', 'model_file', 'use_slap', 'cc_fn','dropout', 'baseline_weights'])
    
    for i in range(len(model_files)):
        start_time = time.time()
        model_file, use_slap, cc_fn, dropout = model_files[i], use_slaps[i], cc_fns[i], dropouts[i]
        print('AI tested:', model_file)
        pool = mp.Pool(cpu_n)
        r = pool.starmap(AI_compete, zip([2]*50, [model_file]*50, [use_slap]*50, [cc_fn]*50, [dropout]*50, [baseline]*50, [is_shown]*50))
        pool.close()
        win_ratio = np.mean(r)
        duration = time.time() - start_time
        with open('./data/'+file_name+'.csv', 'a+', encoding='UTF8', newline='') as f:
            csv.writer(f).writerow([win_ratio, duration, model_file, use_slap, cc_fn, dropout, baseline['model_file']])
        print('AI tested:', model_file)
        print('win ratio:', win_ratio, '  duration: {:.2f} min'.format(duration/60))
    
    if is_jarviscloud:  jarviscloud.pause()    # auto pause cloud instance to avoid overcharge


if __name__ == '__main__':
    cpu_n = 1
    file_name = 'AI_compete_s0(4)_vs_n0(2)'
    is_shown = 1
    randomize()
    # basline config
    baseline = {}
    baseline['model_file'] = './weights/checkpoint_v11id2_n0(2)_Adam_0.25noise0.25_0.3D_0.25expl_0.004lr_0.0001L2_0drp_10000buffer_10e_5000.model'
    baseline['use_slap'] = False
    baseline['cc_fn'] = None
    baseline['dropout'] = 0
    # AIs to be tested, in list
    model_files =[]
    #model_files.append('./weights/checkpoint_v11id2_s0(1)_Adam_0.1noise0.4_0.3D_0.25expl_0.0005lr_0.001L2_0.2drp_1250buffer_10e_5000.model')
    #model_files.append('./weights/checkpoint_v11id2_s0(2)_Adam_0.1noise0.4_0.15D_0expl_0.00025lr_0.0001L2_0.2drp_1250buffer_10e_5000.model')
    #model_files.append('./weights/checkpoint_v11id2_s0(3)_Adam_0.1noise0.4_0.15D_0expl_0.00025lr_0.0001L2_0drp_1250buffer_10e_5000.model')
    model_files.append('./weights/checkpoint_v11id2_s0(4)_Adam_0noise0_0.15D_0expl_0.00025lr_0.001L2_0.2drp_1250buffer_10e_5000.model')
    use_slaps = [False]
    cc_fns = [None]
    dropouts = [0.2]
    
    run(cpu_n, model_files, use_slaps, cc_fns, dropouts, file_name, baseline, is_shown)

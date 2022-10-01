#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 17:35:43 2022

@author: suenchihang
# train_multi2: fix bug: hyerparameters set were not running through as __init__ 
#   has set by previous hyerparameters for some functions, e.g. policy_value_fn
"""

from train_v11id2 import TrainPipeline, randomize
import numpy as np
import random
import torch
from collections import defaultdict, deque
from game_array import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alpha0_reuse import MCTSPlayer
import time
from datetime import date, datetime
import itertools
from slap6 import slap, slap_opening, cc_state, cc_pos, stone_pos
import csv
import os
import sys
import subprocess
import pickle

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

from policy10 import PolicyValueNet   # import after autoclip is installed

def trial_run(use_slap='replace', num_ResBlock=0, learn_rate=1e-3, optimizer='Adam', dropout=0, L2=1e-4, c_puct=5, noise=(0, 0), Dirichlet=0.15, explore=0.25, buffer_size=10000, buffer_size_later=1250, batch_size=512, stage0_duration=100000, epochs_early=10, epochs_full=10, game_batch_num=250, trial_run_name=None):
    randomize()
    t = TrainPipeline()
    t.use_slap = use_slap
    t.num_ResBlock = num_ResBlock
    t.learn_rate = learn_rate
    t.optimizer = optimizer
    t.dropout = dropout
    t.L2 = L2
    t.c_puct = c_puct
    t.noise = noise
    t.Dirichlet = Dirichlet          
    t.explore = explore 
    t.buffer_size =  buffer_size       #auto-scaled down by 8 if use_slap == 'replace'
    t.buffer_size_later = buffer_size_later 
    t.batch_size = batch_size 
    t.stage0_duration = stage0_duration
    t.epochs_early = epochs_early
    t.epochs_full = epochs_full   
    t.game_batch_num = game_batch_num
    lead = ('n' if not use_slap else ('s' if use_slap=='replace' else 'ns'))+str(num_ResBlock)+'_'+optimizer
    tail = '_{}lr_{}drp_{}L2_{}cp_{}noise{}_{}D_{}expl_{}buf'.format(learn_rate, dropout, L2, c_puct, noise[0], noise[1], Dirichlet, explore, buffer_size)
    t.trial_run_name = lead + tail if trial_run_name is None else trial_run_name
    if t.use_slap == 'replace':
        t.buffer_size = int(t.buffer_size/8)     #auto scale down, to avoid too old games
    t.data_buffer = deque(maxlen=t.buffer_size)
    opening = slap_opening(t.board) if t.slap_open else None
    in_channel = 8 if t.cc_fn else 4
    net_height = 2*t.board_height + 1 if t.use_slap == 'add' else t.board_height
    t.policy_value_net = PolicyValueNet(t.board_width, net_height, None, t.use_slap, t.num_ResBlock, t.L2, opening, t.Dirichlet, t.optimizer, t.dropout, t.extra_act_fc, in_channel, t.cc_fn, t.normalized)
    t.mcts_player = MCTSPlayer(t.policy_value_net.policy_value_fn, t.c_puct, t.n_playout, is_selfplay=1, noise=t.noise, explore=t.explore)
    
    assert t.noise[0] <= t.noise[1], f"noise should begin with smaller number, got: {t.noise}"
    assert t.use_slap in ['add', 'replace', False], f"use_slap should be 'add', 'replace' or False, got: {t.use_slap}"
    
    t.config_names = ['script_name', 'keep_old_csv', 'board_width','board_height', 'n_in_row','learn_rate', 'adaptive_n','adaptive_sigma_num', 'dropout', 'extra_act_fc', 'optimizer', 'L2', 'lr_multiplier','temp', 'n_playout', 'c_puct', 'buffer_size', 'buffer_size_later','batch_size', 'play_batch_size', 'stage0_duration','epochs_early', 'epochs_full', 'kl_targ', 'check_freq', 'checkpoint_freq', 'game_batch_num', 'start_i', 'best_win_ratio', 'pure_mcts_playout_num', 'normalized', 'only_multi_evaluate', 'trial_run_name', 'current_policy_path', 'best_policy_path', 'noise', 'Dirichlet', 'explore', 'evaluate_cpu_n', 'num_ResBlock',  'use_slap', 'slap_open', 'warmup_offset', 'cc_fn' ]
    t.config_values = [__file__, t.keep_old_csv, t.board_width, t.board_height, t.n_in_row, t.learn_rate, t.adaptive_n, t.adaptive_sigma_num, t.dropout, t.extra_act_fc, t.optimizer, t.L2, t.lr_multiplier, t.temp, t.n_playout, t.c_puct, t.buffer_size, t.buffer_size_later, t.batch_size, t.play_batch_size, t.stage0_duration, t.epochs_early, t.epochs_full, t.kl_targ, t.check_freq, t.checkpoint_freq, t.game_batch_num, t.start_i, t.best_win_ratio, t.pure_mcts_playout_num, t.normalized, t.only_multi_evaluate, t.trial_run_name, t.current_policy_path, t.best_policy_path, t.noise, t.Dirichlet, t.explore, t.evaluate_cpu_n, t.num_ResBlock, t.use_slap, t.slap_open, t.warmup_offset, t.cc_fn]
    
    t.run()
    return t.trial_run_name, np.mean(t.win_record[-1]), np.mean(t.loss_record[-25:], axis=0)[t.loss_head.index('v_loss')], t.loss_record[-1][t.loss_head.index('loss')], t.total_time, sum(t.game_time), np.mean(t.game_len)
    
    
if __name__ == '__main__':
    experiment_name = 'Train_multi2b_s0_Adam_5cp_1250buf'  #lr: 1e-3, 5e-4, 2.5e-4 drp: 0, 0.2
    timestamp = datetime.now().strftime("_%d%b%Y-%H_%M_%S")   #skip timestamp if want to append old file
    skip_timestamp = False    #skip timestamp if want to append old file
    file = './data/'+ experiment_name + (timestamp if not skip_timestamp else '') +'.csv'
    trial_record =[]
    
    if not os.path.isfile(file):
        with open(file, 'w', encoding='UTF8', newline='') as f:
            csv.writer(f).writerow(['trial_name', 'win_ratio', 'validation_loss', 'train_loss', 'total_time', 'selfplay_time', 'game_len', 'use_slap', 'num_ResBlock', 'optimizer', 'buffer_size', 'c_puct', 'learn_rate', 'dropout', 'L2', 'explore', 'noise', 'Dirichlet'])
    
    use_slap ='replace'
    num_ResBlock = 0
    optimizer ='Adam'
    buffer_size = 10000    #auto-scaled down by 8 if use_slap == 'replace'
    c_puct = 5    
    
    learn_rate = 5e-4     #lr: 1e-3, 5e-4, 2.5e-4 drp: 0, 0.2
    dropout = 0
    print('lr:', learn_rate, ' dropout:', dropout, experiment_name)
    
    for L2 in [1e-3, 1e-4]:
        for explore in [0, 0.25]:
            for noise in [(0, 0), (0.25, 0.25), (0.1, 0.4)]:
                for Dirichlet in [0.15, 0.3]:
                    if not( noise==(0,0) and Dirichlet == 0.3):
                        trial_data = trial_run(use_slap=use_slap, num_ResBlock=num_ResBlock, learn_rate=learn_rate, optimizer=optimizer, dropout=dropout, L2=L2, c_puct=c_puct, noise=noise, Dirichlet=Dirichlet, explore=explore, buffer_size=buffer_size)
                        trial_data = trial_data +(use_slap, num_ResBlock, optimizer, buffer_size, c_puct, learn_rate, dropout, L2, explore, noise, Dirichlet)
                        trial_record.append(trial_data)
                        with open(file, 'a+', encoding='UTF8', newline='') as f:
                            csv.writer(f).writerow(trial_data)
    
    if is_jarviscloud:  jarviscloud.pause()    # auto pause cloud instance to avoid overcharge

    
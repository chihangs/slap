#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:24:13 2022
use synthetic states to test diff neural network performance for tuning
# version 4a: config to test SLAP & data augmentation by diff seeds, change model name
# version 4: allow cc_state to be tested: use slap6b, in_channel=8; update file columns
    remove extra datasets; save_weight = True, change model_name & config
# version 3b: create diff datasets & tailor-made model name for experiment tuning dataset; add randomize
# version 3a: change config: num_random=1000, win_set=1, Adam s0, n0, lr around 0.0001, L2 1e-4, 0 dropout, add auto-pause
# version3 uses policy7 & train_v9l to simplify & use optimizer, try AdamW
# version2 keeps the same function train_validate unchanged except adding display for extra_act_fc; 
  script in main body is changed for fine tune stage 2
@author: suenchihang
"""

from train_v9l import TrainPipeline
from synthetic import synthetic_states
from game_array2trial import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alpha0_reuse import MCTSPlayer
from policy7 import PolicyValueNet
import time
from datetime import date, datetime
import itertools
from slap6b import slap, slap_variants, slap_opening, unslap, bbox, centre, bbox_colour, cc, cc_state
import csv
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


def train_validate(use_slap, Num_ResBlock, dataset, epochs, batch_size, L2, optimizer, dropout, lr, check_freq, extra_act_fc=0, file=None, save_weight=False, cc_fn=None, noAugm=False):
    t = TrainPipeline()
    t.use_slap = use_slap
    t.learn_rate = lr
    special = ('_cc_state' if cc_fn else '') + ('_noAugm' if noAugm else '')
    t.data_buffer = dataset[('Slap' if use_slap else 'noSlap')+ special +'_train']
    size = len(t.data_buffer)
    #model_name = ('s' if use_slap else 'n')+'{}_{}lr_'.format(Num_ResBlock, lr)+optimizer+'{}dropout_{}L2'.format(dropout, L2)+('_extra' if extra_act_fc else '')+'_'+str(batch_size)+'b'
    model_name = str(L2)+'L2'+special+'_'+('Slap' if use_slap else 'noSlap')
    test_data = dataset[('Slap' if use_slap else 'noSlap')+ special +'_test']
    t.epochs = check_freq
    t.policy_value_net = PolicyValueNet(8, 8, None, use_slap, Num_ResBlock, L2, optimizer=optimizer, dropout=dropout, extra_act_fc=extra_act_fc, in_channel=8 if cc_fn else 4)
    print('---------------------------------')
    print('use_slap:{}, epochs: {}, batch_size: {}, num_Res: {}, extra_act_fc: {}, dropout: {}, lr: {}, optimizer: {}, L2: {},'.format(use_slap, epochs, batch_size, Num_ResBlock, extra_act_fc, dropout, lr, optimizer, L2))
    print('Training dataset size: ', len(t.data_buffer))
    print('Validation dataset size: ', len(test_data))
    start = time.time()
    for i in range(int(epochs/check_freq)):
        train_result = t.policy_update(sample_size=train_batch_size, validation=False)
        state_batch, mcts_probs, winner_batch = zip(*test_data)
        with torch.no_grad():
            loss, entropy, value_loss, policy_loss = t.policy_value_net.train_step(state_batch, mcts_probs, winner_batch, lr, train_mode=False)
        print('Validation:')
        print('value_loss:{:.3f}, policy_loss:{:.3f}, loss:{:.5f}, entropy:{:.5f}'.format(value_loss, policy_loss, loss, entropy))
        if file:
            with open(file, 'a+', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_name, loss, entropy, value_loss, policy_loss]+[use_slap, Num_ResBlock, lr, optimizer, dropout, L2, extra_act_fc, batch_size, special]+train_result)
    if save_weight:
        t.policy_value_net.save_model('./weights/synthetic_'+model_name+'_'+str(epochs)+'.model')
    print('time: {:.2f} sec'.format(time.time()-start))
    return loss, entropy, value_loss, policy_loss, model_name


def randomize(seed_rng=123, seed_np=123, seed_torch=123):
    os.environ['PYTHONHASHSEED'] = '0'
    rng = np.random.default_rng(seed_rng)
    np.random.seed(seed_np)
    torch.manual_seed(seed_torch)


def add_cc(state, cc_fn=cc_state):
    cc_info = cc_fn(state)
    return np.concatenate((state, cc_info))


def get_add_cc(play_data):
    extend_data = []
    for state, mcts_prob, winner in play_data:
        extend_data.append((add_cc(state), mcts_prob, winner))
    return extend_data


if __name__ == '__main__':
    randomize(42, 42, 42)
    t = TrainPipeline()
    save_weight = True

    train_epochs = 10000
    check_freq = 10
    train_batch_size = 512

    dataset = {}
    data = synthetic_states(num_random=8000, win_set=8)
    data_train, data_test = train_test_split(data, test_size=0.15)  #10064, 1776
    dataset['noSlap_test'] = data_test
    dataset['Slap_test'] = t.get_slap_data(data_test)
    dataset['noSlap_train'] = t.get_equi_data(data_train)
    dataset['Slap_train'] = t.get_slap_data(data_train)
    dataset['noSlap_noAugm_test'] = data_test
    dataset['noSlap_noAugm_train'] = data_train
    dataset['noSlap_cc_state_test'] = get_add_cc(data_test)
    dataset['noSlap_cc_state_train'] = get_add_cc(data_train)
   

    trial_name = 'Tune_SLAP_noSLAP_10000e_seed42'
    timestamp = datetime.now().strftime("_%d%b%Y-%H:%M:%S")    
    file = './data/'+ trial_name + timestamp +'.csv'
    
    with open(file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model','loss','entropy','value_loss','policy_loss', 'use_slap', 'Num_ResBlock', 'lr', 'optimizer', 'dropout', 'L2', 'extra_act_fc', 'batch_size', 'special', 'train_loss', 'train_entropy', 'train_value_loss', 'train_policy_loss', 'kl', 'lr_multiplier', 'explained_var_old', 'explained_var_new'])

    min_loss = 10000
    best_model = None
    
    optimizer = 'Adam'
    use_slap = False
    Num_ResBlock = 0
    extra_act_fc = False
    L2 = 1e-4
    dropout = 0
    lr = 1e-3
    cc_fn = None
    noAugm = False
    
    for use_slap in [True , False]:
        for L2 in [1e-3, 1e-4, 1e-5]:
                loss, _, _, _, model = train_validate(use_slap, Num_ResBlock, dataset, train_epochs, train_batch_size, L2, optimizer, dropout, lr, check_freq, extra_act_fc, file, save_weight, cc_fn, noAugm)
                if loss < min_loss:
                    min_loss = loss
                    best_model = model
    
    print('best model: ', best_model, 'loss:', min_loss)
    with open(file, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['best_model:'+best_model, min_loss])
        
    from jarviscloud import jarviscloud       # auto pause cloud instance at end to avoid overcharge
    jarviscloud.pause()
                



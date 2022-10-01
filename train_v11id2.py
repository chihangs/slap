# -*- coding: utf-8 -*-
"""  An implementation of the training pipeline of AlphaZero for Gomoku
Adapted from https://github.com/junxiaosong/AlphaZero_Gomoku
Modified by Chi-Hang Suen:

# v11id2: fix bug on keep_old vs create file, save buffer also at checkpoint 
#         add self.start_time, self.total_time, self.win_record
# v11id: use back policy10 & game_array & slap6 in v11d (also remove baseline_transform), but keep others in v11i    
# v11i: use policy11d & game_slap3 to fix bug (suspect to happen from v11e);
#      self.baseline_transform to choose transforming state before inference in baseline case & add baseline_transform & keep_old_csv in report
# v11h: game_slap2 to fix bug, policy11c for refactoring 
#       save data_buffer and load initial data_buffer (if any)
#       new_file - don't create if already exisits and if self.keep_old_csv
# v11g: import policy11b: fix bug in random.sample (doesn't affect slap-replace models already running)
#       add is_jarviscloud in result file     
# v11f: fix time conversion bug in checkpoint,
#       policy11a imported to follow AlphaGo Zero to transform randomly to one 
#                variant for net inference & fix bug in act_probs for add slap 
# v11e: import policy11, game_slap & slap7 to pre-compute slap variants and speed up
# v11d: print script name & trial name at start; log config hyperparameters in result; 
#       don't change self.buffer_size in place, direclty use self.buffer_size_later for 2nd stage
# v11c: allow config n and sigma_num of adaptively decreasing lr
# v11b: decrease learning rate when validation loss increases by 2 sigma; add randomize
# v11a: try import jarviscloud first; cater for Hyperion to install autoclip; allow config buffer size in second stage
# v11: import policy10 to add use_slap usage: slap as supplement instead of replacement of data augmentation
#      config self.use_slap as 'add', 'replace' or False
# v10j: import policy9b to fix bug (miss reshape for unslap in one upgrade)!!!
# v10i: fix episode_len when play_batch_size > 1, and rename self.epochs
# v10h4: config and change early stage timing
# v10h3: config epochs at 2 stages
# v10h2: fix epoch issue in policy_update due to resume of full buffer size
# v10h: cancel buffer auto scale and resume full buffer size at 100 games
# v10g: import policy9a to use auto clip instead; auto create data & weights directory
# v10f: when data_buffer is not full, train fewer epochs in policy_update
# v10e: use policy9 to clip neural net gradients & add option to normalize or not the probs before MCTS
# v10d: fix bug: move the code 'self.data_buffer = deque(maxlen=self.buffer_size)' to after auto-scale
#       note: policy8 still uses slap4 (as cc_fn is fed anyway) while this file uses slap6
# v10c: import slap6 to allow stone_pos in addition to cc_pos
# v10b: import slap 5, allow using cc_pos instead of cc_state
# v10a: add warmup_offset option to choose offseting sample diff of warmup size due to use of slap; remove self.prioritised & self.prioritised_loop
# v10: import slap4 and policy8 to exploit crop & centre transformation in equi_data & slap_data
# v9l: upgrade from v9k_tested; add AdamW option and validation loss via policy7.py, 
#  accordingly change config SGD to optimizer & amend policy_update & run
# v9k_tested: same as v9k in Hyperion or Jarvislabs, but diff from v9k in local computer which uses game_array2trial
# v9k: import policy6b: fix bug on net for extra_act_fc 
# v9j: add option for dropout & extra_act_fc via policy6.py
# v9i: build from v9h2, add SGD optimizer option via policy5.py
# v9h2: try not using adaptive learning rate or early stop in policy_update
# v9h: import policy4: add evaluation mode for batchnorm & use noise after (instead of before) masking out illegal actions, remove noise count & sum
# v9g: allow policy learning before full batch size is obtained; np.mean replaces statistics.mean
# v9f: fix bug to get data buffer
# v9e: import mcts_alpha0_reuse to reuse search sub-tree for evaluation mode as well; 
#      import policy3 to pass value_loss, policy loss for record
# v9d: import mcts_alpha0_explore to allow extra exploration noise after MCTS search, config explore
# v9c: import mcts_alpha0_refill: expand and refill with noise for root if self-play
# v9b: import policy2 to vary noise against leaf value and count noise freq, change noise format
# v9: use game_array which speeds up checking by using the fact that previous state must be non-terminal;
#     use policy to combine slap & non-slap, and allow diff networks used and add corresponding config;
#     use mcts_alpha0 to remove some unnecessary steps
# v8 versions have wrong directions, all discarded
# v7b: remove flip upside down in equi_data & slap_data as current state no longer stored as upside down in game_7b; 
#      hence policy_slap_noise is changed accordingly to policy_slap_noise_v7b
# v7a2: update policy to have consistent output format (for non-slap case), add auto-pause,
#       config evaluate_cpu_n, start_i, show time per move, record time in print_end
# ignore v7.1 - v7.9 trials
# v7 remove parallel process for selfplay_data & print debug
# v6.9 use process for collect_selfplay_data instead of pool for parallel_selfplay_data
# v6.8 set cpu_n =7, use pool.starmap for parallel_selfplay_data (which adds dummy arg)
#v6.7: don't use new train; set cpu_n differently for 2 processes
# v6.6 test: set new train object in parallel_selfplay_data; pool changed back to pool map
# v6.5 test pool map changed to pool map_async for MCTS
# v6.4 change GPU back to 7, set new board and game in policy_evaluate & parallel_selfplay_data
# v6.3 fix bug in extend results, test 64 GPUs
# v6.2 print for debug, change epoch back to 5; align with noise version v5.6, multi eval freq set to 50 for debug
# v6.1 use torch.multiprocessing instead and set spawn as start method 
# v6  parallel programming
# v5.4 fix bug in rot90 axes and fliplr, flipud (use axis instead), fix print_end; auto import policy_slap or not; auto downsize buffer_size by 8 if slap is used
# v5.3 fix dirichlet (original code applied outside simulation, which is wrong): just use mcts_alphaZero_amend, policy_value_net_pytorch_amend; fix early stage: game_amend
# v5.2 record kl, lr_multiplier, adjust final_evaluate and change name to multi_evaluate
# checkpoint_freq changed to 250 and do multi_evaluate
Add slap transformation, time, current time, final_evaluate, update policy by diff batches instead for each iteration, self.prioritised, self.prioritised_loop; add only_final_evaluate, current & best policy path in __init__ , save csv & checkpoint, trial_run_name  """

from __future__ import print_function
import random
import numpy as np
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


def randomize(seed_rng=123, seed_np=123, seed_torch=123):
    os.environ['PYTHONHASHSEED'] = '0'
    rng = np.random.default_rng(seed_rng)
    np.random.seed(seed_np)
    torch.manual_seed(seed_torch)

class TrainPipeline():
    def __init__(self, init_model=None, init_buffer=None):
        # params of the board and the game
        self.keep_old_csv = False     #if true, append to old result & loss (but not game) csv with same trial_run_name
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.learn_rate = 1e-3       # learning rate
        self.adaptive_n = 100        # no of iterations of loss to average to decide adaptively decreasing lr, set large num if don't want adaptively decrease
        self.adaptive_sigma_num = 3  # no of sigma to decide significant increase of loss or not
        self.dropout = 0             # dropout probability in neural net
        self.extra_act_fc = False     # add extra action FC layer or not
        self.optimizer = 'Adam'              #optimizer options: 'SGD', 'Adam', 'AdamW' 
        self.L2 = 1e-4               # coef of L2 penalty
        self.lr_multiplier = 1.0     # adaptively adjust the learning rate
        self.temp = 1.0              # the temperature param
        self.n_playout = 400         # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000      #auto downsize by 8 times if slap is used
        self.buffer_size_later = 1250   #buffer size at second stage
        self.batch_size = 512         # mini-batch size for training
        self.play_batch_size = 1      # no of games in a batch
        self.stage0_duration = 100000    # early stage duration by no of games; enter large num if skip second stage; enter 0 if skip stage0 i.e. want full batch size and full epoch for all stages
        self.epochs_early = 10            # num of train_steps per update at early stage if use_slap
        self.epochs_full = 10             # num of train_steps for each update in second stage if use_slap
        self.kl_targ = 0.02
        self.check_freq = 25           #freq to check against evaluation agent and save model
        self.checkpoint_freq = 250      #freq to create and save checkpoint for weights
        self.game_batch_num = 1500       # batches of games
        self.start_i = 0                #start range of self-play loops, 0 if no init_model, otherwise enter end range of previous training 
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000       # num of simulations for pure mcts (evaluation agent)
        self.normalized = True                # whether to normaize prior probs before MCTS
        self.only_multi_evaluate = True
        self.trial_run_name = 'v11id2_s0_0noise_0.25explore_Adam_0.001lr_1250buffer_10e'
        self.current_policy_path = './weights/current_'+self.trial_run_name+'.model'
        self.best_policy_path =  './weights/best_'+self.trial_run_name+'.model'
        self.noise = (0, 0)     #weight of randomness, e.g.(0.1, 0.4) means weight sliding over 0.1 to 0.4 against leaf value
        self.Dirichlet = 0.15           #Dirichlet noise parameter, to scale against average no of legal moves
        self.explore = 0.25            # extra exploration noise after MCTS search
        self.evaluate_cpu_n = 7                # no of CPU cores allowed
        self.num_ResBlock = 0 
        #self.baseline_transform = True     #AlphaGo Zero: whether to transform state to a random variant for network inference
        self.use_slap  = 'replace'        #'add', 'replace' or False, supplement or replacement of data augmentation
        self.slap_open = False
        self.warmup_offset = 8       # 8: fully offset impact on warmup size due to slap, 1: not to offset; partial if bewteen
        self.cc_fn = None           # None, cc_state, or cc_pos: function to get crop & centre info; stone_pos: just stone positions scaled
        
        if self.use_slap == 'replace':
            self.buffer_size = int(self.buffer_size/8)     #auto scale down, to avoid too old games
        self.data_buffer = deque(pickle.load(open(init_buffer, 'rb')), maxlen=self.buffer_size) if init_buffer else deque(maxlen=self.buffer_size)
        opening = slap_opening(self.board) if self.slap_open else None
        in_channel = 8 if self.cc_fn else 4
        net_height = 2*self.board_height + 1 if self.use_slap == 'add' else self.board_height
        self.policy_value_net = PolicyValueNet(self.board_width, net_height, init_model, self.use_slap, self.num_ResBlock, self.L2, opening, self.Dirichlet, self.optimizer, self.dropout, self.extra_act_fc, in_channel, self.cc_fn, self.normalized)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, self.c_puct, self.n_playout, is_selfplay=1, noise=self.noise, explore=self.explore)
        
        assert self.noise[0] <= self.noise[1], f"noise should begin with smaller number, got: {self.noise}"
        assert self.use_slap in ['add', 'replace', False], f"use_slap should be 'add', 'replace' or False, got: {self.use_slap}"
        
        self.config_names = ['script_name', 'keep_old_csv', 'board_width','board_height', 'n_in_row','learn_rate', 'adaptive_n','adaptive_sigma_num', 'dropout', 'extra_act_fc', 'optimizer', 'L2', 'lr_multiplier','temp', 'n_playout', 'c_puct', 'buffer_size', 'buffer_size_later','batch_size', 'play_batch_size', 'stage0_duration','epochs_early', 'epochs_full', 'kl_targ', 'check_freq', 'checkpoint_freq', 'game_batch_num', 'start_i', 'best_win_ratio', 'pure_mcts_playout_num', 'normalized', 'only_multi_evaluate', 'trial_run_name', 'current_policy_path', 'best_policy_path', 'noise', 'Dirichlet', 'explore', 'evaluate_cpu_n', 'num_ResBlock',  'use_slap', 'slap_open', 'warmup_offset', 'cc_fn' ]
        self.config_values = [__file__, self.keep_old_csv, self.board_width, self.board_height, self.n_in_row, self.learn_rate, self.adaptive_n, self.adaptive_sigma_num, self.dropout, self.extra_act_fc, self.optimizer, self.L2, self.lr_multiplier, self.temp, self.n_playout, self.c_puct, self.buffer_size, self.buffer_size_later, self.batch_size, self.play_batch_size, self.stage0_duration, self.epochs_early, self.epochs_full, self.kl_targ, self.check_freq, self.checkpoint_freq, self.game_batch_num, self.start_i, self.best_win_ratio, self.pure_mcts_playout_num, self.normalized, self.only_multi_evaluate, self.trial_run_name, self.current_policy_path, self.best_policy_path, self.noise, self.Dirichlet, self.explore, self.evaluate_cpu_n, self.num_ResBlock, self.use_slap, self.slap_open, self.warmup_offset, self.cc_fn]
        
    def add_cc(self, state):  
        """" state: current_state of board
            return: concat crop & centre info with scaled position index; 
        if can't be exact centre, slightly baised towards top left"""
        cc_info = self.cc_fn(state)
        return np.concatenate((state, cc_info))
    
    def get_equi_data(self, play_data):
        """  augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...] """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            if self.use_slap == 'add':
                slap_state, temp_flip, temp_i = slap(state)
                pi = mcts_prob.reshape(self.board_height, self.board_width)/2  #algin with state shape; 1/2 for weight of slap
                if temp_flip == 'flip':
                    pi = np.flip(pi, axis=-1)
                pi = np.rot90(pi, temp_i, axes=(-2, -1))
            for i in [0, 1, 2, 3]:
                # rotate anti-clockwise
                equi_state = np.array([np.rot90(s, i, axes=(-2, -1)) for s in state])
                equi_mcts_prob = np.rot90(mcts_prob.reshape(self.board_height, self.board_width), i, axes=(-2, -1))
                for flip in [False, True]:  #oder of False/True matters
                    if flip:    # flip horizontally by axis=-1, then flatten for mcts_prob shape
                        equi_state = np.array([np.flip(s,axis=-1) for s in equi_state])
                        equi_mcts_prob = np.flip(equi_mcts_prob, axis=-1)
                    if self.cc_fn:    # don't add cc to equi_state directly, as flip & cc not commutable
                        extend_data.append((self.add_cc(equi_state), equi_mcts_prob.flatten(), winner))
                    else:
                        if self.use_slap == 'add':
                            current_state = np.concatenate((equi_state, np.zeros((4,1,self.board_width)), slap_state), axis=-2)
                            current_mcts_prob = np.concatenate((equi_mcts_prob/2, np.zeros((1, self.board_width)), pi), axis=-2)
                            extend_data.append((current_state, current_mcts_prob.flatten(), winner))
                        else:
                            extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
        return extend_data

    def get_slap_data(self, play_data):
        """    augment the data set by slap
        play_data: [(state, mcts_prob, winner_z), ..., ...]  """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            slap_state, temp_flip, temp_i = slap(state)
            pi = mcts_prob.reshape(self.board_height, self.board_width) #algin with state shape
            if temp_flip == 'flip':
                pi = np.flip(pi, axis=-1)
            pi = np.rot90(pi, temp_i, axes=(-2, -1))
            mcts_prob = pi.flatten()    # back to move representation of board 
            if self.cc_fn:
                slap_state = self.add_cc(slap_state)
            extend_data.append((slap_state, mcts_prob, winner))
        return extend_data
    
    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""    #amend for parallel processing
        episodes_len = []
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            episodes_len.append( len(play_data) )
            # augment the data
            play_data = self.get_slap_data(play_data) if self.use_slap == 'replace' else self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
        self.episode_len = np.mean(episodes_len)
        
    def policy_update(self, sample_size=512, validation=True, epochs=5):
        """update the policy-value net"""
        if validation:
            with torch.no_grad():
                validation_batch =  list(itertools.islice(self.data_buffer, max(self.data_len -int(self.episode_len*self.play_batch_size), 0), self.data_len))
                state_batch, mcts_probs_batch, winner_batch = zip(*validation_batch)
                v_loss, v_entropy, v_value_loss, v_policy_loss = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, 0, train_mode=False)
        else:
            v_loss, v_entropy, v_value_loss, v_policy_loss = None, None, None, None
        
        n = self.adaptive_n  # no of iterations to average
        if len(self.loss_record)%n == 0 and len(self.loss_record) >= n*2:
            v_loss_idx = self.loss_head.index('v_loss')
            previous_mean = np.mean(self.loss_record[-n*2:-n], axis=0)[v_loss_idx]
            current_mean = np.mean(self.loss_record[-n:], axis=0)[v_loss_idx]
            sigma = np.std(self.loss_record[-n:], axis=0)[v_loss_idx]/(n**0.5)
            if current_mean - previous_mean > self.adaptive_sigma_num * sigma:   # ~ 99% confidence for 3 sigma
                self.lr_multiplier /=2         # adaptively decrease lr
        
        for i in range(epochs):
            mini_batch = random.sample(self.data_buffer, sample_size) 
            state_batch, mcts_probs_batch, winner_batch = zip(*mini_batch)
            old_probs, old_v = self.policy_value_net.policy_value(state_batch)
            loss, entropy, value_loss, policy_loss = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),axis=1))

        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        print(("kl:{:.5f}, lr_multiplier:{:.3f}, validation_loss:{:.3f}, loss:{:.3f}, entropy:{:.3f}, explained_var_old:{:.3f}, explained_var_new:{:.3f}").format(kl, self.lr_multiplier, v_loss, loss, entropy, explained_var_old, explained_var_new))
        return [loss, entropy, value_loss, policy_loss, kl, self.lr_multiplier, explained_var_old, explained_var_new, v_loss, v_entropy, v_value_loss, v_policy_loss]

    def policy_evaluate(self, n_games=10, pure_n_playout=None):
        """  Evaluate the trained policy by playing against the pure MCTS player
                Note: this is only for monitoring the progress of training       """
        if pure_n_playout is None:
            pure_n_playout = self.pure_mcts_playout_num
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, self.c_puct, self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=pure_n_playout)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            new_board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
            new_game = Game(new_board)
            winner = new_game.start_play(current_mcts_player, pure_mcts_player, start_player=i % 2, is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(pure_n_playout,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def multi_evaluate(self, n_games=10):
        start = time.time()
        win_ratios=[]
        self.pure_mcts_playout_num = 1000
        for i in range(3):
            win_ratios.append(self.policy_evaluate(n_games))
            self.pure_mcts_playout_num += 2000
        print('Overall win ratio (tie treated as half win): {:.3f}'.format(np.mean(win_ratios)))
        e_time = time.time() - start
        print('Time for multi evaluation: {:.2f} minutes'.format((e_time)/60))
        self.write_file('result', win_ratios)
        return e_time
        
    def new_file(self, file_name_head, header, keep_old=False):
        if not( keep_old and os.path.isfile('./data/'+file_name_head+'_'+self.trial_run_name+'.csv') ):
            with open('./data/'+file_name_head+'_'+self.trial_run_name+'.csv', 'w', encoding='UTF8', newline='') as f:
                csv.writer(f).writerow(header)

    def write_file(self, file_name_head, data):
        with open('./data/'+file_name_head+'_'+self.trial_run_name+'.csv', 'a+', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
            
    def time_message(self, since=None, msg=None):    #since: start time of sth
        print(datetime.now().strftime("%d-%b-%Y %H:%M:%S"), msg,'time passed: {:.3f} min'.format((time.time()-since)/60))
   
    def print_end(self):
        self.time_message(since=self.start_time, msg='total')
        self.total_time = time.time() - self.start_time
        print('Time - MCTS: {:.4f} hr; evaluation: {:.4f} hr'.format(sum(self.game_time)/3600, sum(self.e_time)/3600))
        print('Policy update time: {:.4f} hr'.format((self.total_time-sum(self.game_time)-sum(self.e_time))/3600))
        print('Average game length in self play: {:.2f}'.format(np.mean(self.game_len)))      
        self.write_file('result', ['game_time', 'e_time', 'update_time', 'total_time', 'game_len', 'is_jarviscloud'])
        self.write_file('result', [sum(self.game_time)/3600, sum(self.e_time)/3600, (self.total_time-sum(self.game_time)-sum(self.e_time))/3600, self.total_time/3600, np.mean(self.game_len), is_jarviscloud])
        with open('./data/game_'+self.trial_run_name+'.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(list(zip(*list(itertools.islice(self.data_buffer, self.data_len-self.batch_size, self.data_len))))[1])
        self.write_file('result', self.config_names)
        self.write_file('result', self.config_values)
        pickle.dump(self.data_buffer, open('./data/buffer_'+self.trial_run_name+'.pkl', 'wb'))
            
    def run(self):
        """run the training pipeline"""
        self.start_time = time.time()
        print('script: ', __file__, self.trial_run_name,'Training started at ', datetime.now().strftime("%d-%b-%Y %H:%M:%S"))
        try:
            os.makedirs('data', exist_ok=True)
            os.makedirs('weights', exist_ok=True)
            self.game_time, self.game_len, self.loss_record, self.e_time, self.win_record = [], [], [], [], []
            self.loss_head = ['loss', 'entropy', 'value_loss', 'policy_loss', 'kl','lr_multiplier','explained_var_old','explained_var_new', 'v_loss', 'v_entropy', 'v_value_loss', 'v_policy_loss']
            self.new_file('loss', self.loss_head, self.keep_old_csv)
            self.new_file('result',['win_1k','win_3k','win_5k'], self.keep_old_csv)
            epochs = self.epochs_early if self.use_slap == 'replace' else self.epochs_full
            for i in range(self.start_i, int(self.game_batch_num)):
                if (i == self.start_i + self.stage0_duration) and self.use_slap == 'replace':
                    self.data_buffer = deque(self.data_buffer, maxlen=self.buffer_size_later)
                    epochs = self.epochs_full
                game_start_t = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                t = time.time() - game_start_t
                self.game_time.append(t)
                print("batch i:{}, episode_len:{}, {:.2f} sec/move".format(i+1, self.episode_len, t/self.episode_len/self.play_batch_size))
                self.game_len.append(self.episode_len)
                self.data_len=len(self.data_buffer)   
                if self.data_len >= self.batch_size/(self.warmup_offset if self.use_slap == 'replace' else 1):
                    loss_data = self.policy_update(sample_size=min(self.batch_size, self.data_len), epochs=epochs)
                    self.loss_record.append(loss_data)
                    self.write_file('loss', loss_data)
                # save model params, check model performance
                if (i+1) % self.check_freq == 0:
                    self.policy_value_net.save_model(self.current_policy_path)
                    print('Average validation loss: {:.3f}'.format(np.mean(self.loss_record[-self.check_freq:], axis=0)[self.loss_head.index('v_loss')]))
                    print(datetime.now().strftime("%d-%b-%Y %H:%M:%S"), "current_"+self.trial_run_name+".model saved; episodes passed: {}".format((i+1)*self.play_batch_size))
                    print('time passed since start: ', (time.time()-self.start_time)/60, 'min')
                    if not self.only_multi_evaluate:
                        win_ratio = self.policy_evaluate()
                        if win_ratio > self.best_win_ratio:
                            print("New best policy!!!!!!!!")
                            self.best_win_ratio = win_ratio
                            self.policy_value_net.save_model(self.best_policy_path)   # update the best_policy
                            if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000):
                                self.pure_mcts_playout_num += 1000
                                self.best_win_ratio = 0.0
                if (i+1) % self.checkpoint_freq == 0:   #evaluate winning percentage
                    self.policy_value_net.save_model('./weights/checkpoint_'+self.trial_run_name+'_'+str(i+1)+'.model')
                    print('checkpoint_'+self.trial_run_name+'_'+str(i+1)+'.model saved')
                    if self.evaluate_cpu_n > 1:       #parallel processing
                        start_e_time = time.time()
                        pool = mp.Pool(self.evaluate_cpu_n)
                        r = pool.starmap(self.policy_evaluate, zip([1]*30, [1000]*10+[3000]*10+[5000]*10))
                        pool.close()
                        win_ratios = [np.mean(r[:10]), np.mean(r[10:20]), np.mean(r[-10:])]
                        self.win_record.append(win_ratios)
                        self.write_file('result', win_ratios)
                        print('Win ratio (tie treated as half win): {:.3f}'.format(np.mean(win_ratios)))
                        print('Time for evaluation: {:.2f} min'.format((time.time()-start_e_time)/60))
                        self.e_time.append(time.time()-start_e_time)
                        print('time passed since start: ', (time.time()-self.start_time)//3600, 'hr', (time.time()-self.start_time)%3600//60, 'min') 
                    else:
                        self.e_time.append(self.multi_evaluate(n_games=10))
                    pickle.dump(self.data_buffer, open('./data/buffer_'+self.trial_run_name+'.pkl', 'wb'))
            self.print_end()
        
        except KeyboardInterrupt:
            print('\n\rquit')
            self.print_end()

if __name__ == '__main__':
    randomize()
    training_pipeline = TrainPipeline()
    training_pipeline.run()
    if is_jarviscloud:  jarviscloud.pause()    # auto pause cloud instance to avoid overcharge

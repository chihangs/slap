# -*- coding: utf-8 -*-
"""  An implementation of the training pipeline of AlphaZero for Gomoku
Adapted from https://github.com/junxiaosong/AlphaZero_Gomoku
Modified by Chi-Hang Suen:
Add slap transformation, time, current time, final_evaluate, update policy by diff batches instead for each iteration, self.prioritised, self.prioritised_loop; add only_final_evaluate, current & best policy path in __init__ , save csv & checkpoint, trial_run_name  """  
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

from __future__ import print_function
import random
import numpy as np
import torch
from collections import defaultdict, deque
from game_array import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alpha0_reuse import MCTSPlayer
from policy7 import PolicyValueNet
import time
from datetime import date, datetime
import itertools
from slap3 import slap, slap_opening
import csv
import os

import torch.multiprocessing as mp
try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass

class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.learn_rate = 2e-3       # learning rate
        self.dropout = 0             # dropout probability in neural net
        self.extra_act_fc = False     # add extra action FC layer or not
        self.optimizer = 'Adam'              #optimizer options: 'SGD', 'Adam', 'AdamW' 
        self.L2 = 1e-4               # coef of L2 penalty
        self.lr_multiplier = 1.0     # adaptively adjust the learning rate based on KL
        self.temp = 1.0              # the temperature param
        self.n_playout = 400         # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000     #auto downsize by 8 times if slap is used
        self.batch_size = 512        # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 10             # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50           #freq to check against evaluation agent and save model
        self.checkpoint_freq = 250      #freq to create and save checkpoint for weights
        self.game_batch_num = 3000
        self.start_i = 1500                #start range of self-play loops, 0 if no init_model, otherwise enter end range of previous training 
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000       # num of simulations for pure mcts (evaluation agent)
        self.only_multi_evaluate = True
        self.trial_run_name = 'v9k_s0_0noise_0.25explore_0.3Dirichlet_10e_Adam'
        self.current_policy_path = './weights/current_'+self.trial_run_name+'.model'
        self.best_policy_path =  './weights/best_'+self.trial_run_name+'.model'
        self.prioritised = False
        self.prioritised_loop = 1         #no of loops to only use most recent games
        self.noise = (0, 0)     #weight of randomness, e.g.(0.1, 0.4) means weight sliding over 0.1 to 0.4 against leaf value
        self.Dirichlet = 0.3           #Dirichlet noise parameter, to scale against average no of legal moves
        self.explore = 0.25            # extra exploration noise after MCTS search
        self.evaluate_cpu_n = 7                # no of CPU cores allowed
        self.num_ResBlock = 0 
        self.use_slap  = True
        self.slap_open = False
        if self.use_slap:
            self.buffer_size = int(self.buffer_size/8)     #auto scale down, to avoid too old games
        opening = slap_opening(self.board) if self.slap_open else None
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, init_model, self.use_slap, self.num_ResBlock, self.L2, opening, self.Dirichlet, self.optimizer, self.dropout, self.extra_act_fc)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, self.c_puct, self.n_playout, is_selfplay=1, noise=self.noise, explore=self.explore)
                          
        assert self.noise[0] <= self.noise[1], f"noise should begin with smaller number, got: {self.noise}"
            
    def get_equi_data(self, play_data):
        """  augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...] """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i, axes=(-2, -1)) for s in state])
                equi_mcts_prob = np.rot90(mcts_prob.reshape(self.board_height, self.board_width), i, axes=(-2, -1))
                extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
                # flip horizontally by axis=-1, then vertically for mcts_prob shape
                equi_state = np.array([np.flip(s,axis=-1) for s in equi_state])
                equi_mcts_prob = np.flip(equi_mcts_prob, axis=-1)
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
            extend_data.append((slap_state, mcts_prob, winner))
        return extend_data
    
    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""    #amend for parallel processing
        for i in range(n_games):
            new_board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
            new_game = Game(new_board)
            winner, play_data = new_game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_slap_data(play_data) if self.use_slap else self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
        
    def policy_update(self, sample_size=512, validation=True):
        """update the policy-value net"""
        if validation:
            validation_batch =  list(itertools.islice(self.data_buffer, self.data_len - self.episode_len, self.data_len))
            state_batch, mcts_probs_batch, winner_batch = zip(*validation_batch)
            v_loss, v_entropy, v_value_loss, v_policy_loss = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, 0, train_mode=False)
        else:
            v_loss, v_entropy, v_value_loss, v_policy_loss = None, None, None, None
        for i in range(self.epochs):
            mini_batch = random.sample(self.data_buffer, sample_size) if i >=self.prioritised_loop or not self.prioritised else list(itertools.islice(self.data_buffer, self.data_len - sample_size, self.data_len))
            state_batch, mcts_probs_batch, winner_batch = zip(*mini_batch)
            old_probs, old_v = self.policy_value_net.policy_value(state_batch)
            loss, entropy, value_loss, policy_loss = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),axis=1))

        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        print(("kl:{:.5f}, lr_multiplier:{:.3f}, loss:{:.5f}, entropy:{:.5f}, explained_var_old:{:.3f}, explained_var_new:{:.3f}").format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))
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
        
    def new_file(self, file_name_head, header):
        with open('./data/'+file_name_head+'_'+self.trial_run_name+'.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def write_file(self, file_name_head, data):
        with open('./data/'+file_name_head+'_'+self.trial_run_name+'.csv', 'a+', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
            
    def time_message(self, since=None, msg=None):    #since: start time of sth
        print(datetime.now().strftime("%d-%b-%Y %H:%M:%S"), msg,'time passed: {:.3f} min'.format((time.time()-since)/60))
   
    def print_end(self):
        self.time_message(since=start_time, msg='total')
        print('Time - MCTS: {:.4f} hr; evaluation: {:.4f} hr'.format(sum(self.game_time)/3600, sum(self.e_time)/3600))
        print('Policy update time: {:.4f} hr'.format((time.time()-start_time-sum(self.game_time)-sum(self.e_time))/3600))
        print('Average game length in self play: {:.2f}'.format(np.mean(self.game_len)))      
        self.write_file('result', ['game_time', 'e_time', 'update_time', 'total_time', 'game_len'])
        self.write_file('result', [sum(self.game_time)/3600, sum(self.e_time)/3600, (time.time()-start_time-sum(self.game_time)-sum(self.e_time))/3600, (time.time()-start_time)/3600, np.mean(self.game_len)])
        with open('./data/game_'+self.trial_run_name+'.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(list(zip(*list(itertools.islice(self.data_buffer, self.data_len-self.batch_size, self.data_len))))[1])
                   
    def run(self):
        """run the training pipeline"""
        print('Training started at ', datetime.now().strftime("%d-%b-%Y %H:%M:%S"))
        try:
            self.game_time, self.game_len, self.loss_record, self.e_time = [], [], [], []
            loss_head = ['loss', 'entropy', 'value_loss', 'policy_loss', 'kl','lr_multiplier','explained_var_old','explained_var_new', 'v_loss', 'v_entropy', 'v_value_loss', 'v_policy_loss']
            self.new_file('loss', loss_head)
            self.new_file('result',['win_1k','win_3k','win_5k'])
            for i in range(self.start_i, int(self.game_batch_num)):
                game_start_t = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                t = time.time() - game_start_t
                self.game_time.append(t)
                print("batch i:{}, episode_len:{}, {:.2f} sec/move".format(i+1, self.episode_len, t/self.episode_len))
                self.game_len.append(self.episode_len)
                self.data_len=len(self.data_buffer)   
                if self.data_len >= self.batch_size/(8 if self.use_slap else 1):
                    loss_data = self.policy_update(sample_size=min(self.batch_size, self.data_len))
                    self.loss_record.append(loss_data)
                    self.write_file('loss', loss_data)
                # save model params, check model performance
                if (i+1) % self.check_freq == 0:
                    self.policy_value_net.save_model(self.current_policy_path)
                    print('Average validation loss: {:.3f}'.format(np.mean(self.loss_record[-self.check_freq:], axis=0)[loss_head.index('v_loss')]))
                    print(datetime.now().strftime("%d-%b-%Y %H:%M:%S"), "Weights saved; episodes passed: {}".format(i+1))
                    print('time passed since start: ', (time.time()-start_time)/60, 'min')
                    if not self.only_multi_evaluate:
                        win_ratio = self.policy_evaluate()
                        if win_ratio > self.best_win_ratio:
                            print("New best policy!!!!!!!!")
                            self.best_win_ratio = win_ratio
                            # update the best_policy
                            self.policy_value_net.save_model(self.best_policy_path)
                            if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000):
                                self.pure_mcts_playout_num += 1000
                                self.best_win_ratio = 0.0
                if (i+1) % self.checkpoint_freq == 0:
                    self.policy_value_net.save_model('./weights/checkpoint_'+self.trial_run_name+'_'+str(i+1)+'.model')
                    print('checkpoint_'+self.trial_run_name+'_'+str(i+1)+'.model saved')
                    if self.evaluate_cpu_n > 1:       #parallel processing
                        start_e_time = time.time()
                        pool = mp.Pool(self.evaluate_cpu_n)
                        r = pool.starmap(self.policy_evaluate, zip([1]*30, [1000]*10+[3000]*10+[5000]*10))
                        pool.close()
                        win_ratios = [np.mean(r[:10]), np.mean(r[10:20]), np.mean(r[-10:])]
                        self.write_file('result', win_ratios)
                        print('Win ratio (tie treated as half win): {:.3f}'.format(np.mean(win_ratios)))
                        print('Time for evaluation: {:.2f} min'.format((time.time()-start_e_time)/60))
                        self.e_time.append(time.time()-start_e_time)
                        print('time passed since start: ', (time.time()-start_time)/60, 'min') 
                    else:
                        self.e_time.append(self.multi_evaluate(n_games=10))
            self.print_end()
        
        except KeyboardInterrupt:
            print('\n\rquit')
            self.print_end()

if __name__ == '__main__':
    start_time=time.time()
    training_pipeline = TrainPipeline()
    training_pipeline.run()
    from jarviscloud import jarviscloud       # auto pause cloud instance at end to avoid overcharge
    jarviscloud.pause()

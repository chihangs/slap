# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the h,w format: 2,3
Adapted from https://github.com/junxiaosong/AlphaZero_Gomoku
Modified by Chi-Hang Suen:
Add play_AI and allow config to play against AI or not, state computer's last move in UI

# v12: align with train_v12 to use game_array3 to fix bug checking win status around diagonal corner
# v11id4: update files to algin with train_v11id4; policy10a instead of policy10 to allow model loaded by CPU
# v8.4 use game_fast4 & mcts_alphaZero_noise3, 
# v8.2 use game_fast2.py (fix bug in game_fast.py) & update other file names, adjust config to fit, e.g. num_ResBlock
# v6: let start_player be config earlier
"""

from __future__ import print_function
import pickle
from game_array3 import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alpha0_reuse import MCTSPlayer
from policy10a import PolicyValueNet



class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            if board.last_move > -1:
                h, w = board.move_to_location(board.last_move)
                print('Move of computer: ',h,',',w,'    ', sep='', end='')
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run_game(model_file = './weights/checkpoint_v11id2_n0(2)_Adam_0.25noise0.25_0.3D_0.25expl_0.004lr_0.0001L2_0drp_10000buffer_10e_5000.model', use_slap=False, cc_fn=None, dropout=0, computer_first=1):
    """
    computer_first: 0 for human first, 1 for computer first
    Further Config here
    """
    
    n = 5   #n in a row to win
    width, height = 8, 8
    play_AI = True     #False means playing against pure MCTS
    num_ResBlock=0
    in_channel = 8 if cc_fn else 4
    alpha=0.3 #not affect game as noise is only applied in self-play
    L2=1e-4   #not affect game as it only affects optimizer in self-play
    
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################

        best_policy = PolicyValueNet(width, height, model_file, use_slap, num_ResBlock, L2, None, alpha, 'Adam', dropout, 0, in_channel, cc_fn) 
        
        if play_AI:   # set larger n_playout if you want better performance
            mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)  
        else:
            mcts_player = MCTS_Pure(c_puct=5, n_playout=5000)

        # human player, input your move in the format: 2,3
        human = Human()
        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=computer_first, is_shown=1)
        
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run_game()

# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/junxiaosong/AlphaZero_Gomoku
Modified by Chi-Hang Suen:
# game_array2trial: add winning_move for experiment with synthetic states
# game_array2: pre-compute state and store, and speed up winner checking
# game_array: speed up checking end game by using the fact that previous state must be non-terminal 
# 7b: use array view instead of coordinates view for move - current state no need to flip upside down, 
#     and tidy up graphics for relevant changes; fix bug mixing up width & height
#start_self_play: align with paper to set temperature at 1 at early stage
"""

from __future__ import print_function
import numpy as np


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2
        
    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be less than {}'.format(self.n_in_row))
        self.black_player = self.players[start_player]  # start player
        self.current_player = self.black_player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
        self.state_colour = np.zeros((2, self.height, self.width))  #first plane is black, white second
        self.pre_computed = False
        self.pre_computed_state = None

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        0 1 2
        3 4 5
        6 7 8
        and move 6's location is (2, 0)
        """
        h = move // self.width
        w = move % self.width
        return (h, w)            # amend as (), better than []

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        if not self.pre_computed:
            self.pre_computed_state = np.zeros((4, self.height, self.width))
            if self.states:
                self.pre_computed_state[0] = self.state_colour[1 - int(self.current_player==self.black_player)]
                self.pre_computed_state[1] = self.state_colour[int(self.current_player==self.black_player)]
                # indicate the last move location
                self.pre_computed_state[2][self.last_move//self.width, self.last_move%self.width] = 1.0
            if len(self.states) % 2 == 0:
                self.pre_computed_state[3][:, :] = 1.0  # indicate the colour to play
            self.pre_computed = True
        
        return self.pre_computed_state

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        h, w = self.move_to_location(move)
        self.state_colour[1 - int(self.current_player==self.black_player)][h, w]= 1
        self.pre_computed = False
        self.current_player = 3 - self.current_player   #change player
        self.last_move = move
       
    def connected_n(self, move_rc, state_colour, n, n_target=None):
        #only check connection with one move in (r,c) format
        if n_target is None:  
            n_target = n
        r, c = move_rc
        s = state_colour
        f = np.fliplr(s)    #flip for finding diagonal in opposite direction, i.e. upwards
        c_f = self.width -1 - c      #for use in flipped state f
        for i in range(n):
             if s[r-i:r+n-i, c].sum() == n_target:  # vertical line
                return True
             if s[r, c-i:c+n-i].sum() == n_target:  #horizontal line
                return True
             if s[r-i:r+n-i, c-i:c+n-i].diagonal().sum() == n_target: #diagonal line
                return True
             if f[r-i:r+n-i, c_f-i:c_f+n-i].diagonal().sum() == n_target: #diagonal line in opposite direction
                return True
        return False

    def has_a_winner(self):
        '''only check connection with last move because state before last move must be non-terminal'''
        n = self.n_in_row
        if self.width * self.height - len(self.availables) < n*2 -1:
            return False, -1
        r, c = self.move_to_location(self.last_move)
        s = self.state_colour[int(self.current_player==self.black_player)] #get state_colour of last move
        has_winner = self.connected_n((r, c), s, n)
        winner = self.states[self.last_move] if has_winner else -1  #note: last move is opponent
        return has_winner, winner
    
    def winning_move(self, s=None, availables=None):
        ''' find immediate winning move before doing a move; s: state_colour 
            for experiment with synthetic states '''
        win_moves=[]
        if s is None:
            s = self.state_colour[1 - int(self.current_player==self.black_player)] #get state_colour of current player
        if availables is None:
            availables = self.availables
        for move in availables:
            if self.connected_n(self.move_to_location(move), s, self.n_in_row):
                win_moves.append[move]
        return win_moves

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        act_temp = 1  #for early stage of game
        while True:
            if len(self.board.availables) < 0.9*self.board.width*self.board.height: 
                act_temp = temp      #use temp if not early stage
            move, move_probs = player.get_action(self.board, act_temp, return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

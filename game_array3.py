# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/junxiaosong/AlphaZero_Gomoku
Modified by Chi-Hang Suen:
# game_array3: fix from game_array (not game_array2trial, which belongs to diff branch)
#              has_a_winner has bug in counting diagonal out of board issue, affect corner case
# speed up checking end game by using the fact that previous state must be non-terminal 
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
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

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
        return [h, w]

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

        square_state = np.zeros((4, self.height, self.width))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.width] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.width] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.width] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        '''only check connection with last move because state before last move must be non-terminal'''
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row
        max_len = self.width * self.height

        if max_len - len(self.availables) < n*2 -1:
            return False, -1

        m = self.last_move
        h, w = self.move_to_location(m)
        player = states[m]
        
        for s in range(max(m-n+1, m-w), min(m, m+width-w-n)+1):       #avoid out of same row; s: starting point
            if len(set(states.get(i, -1) for i in range(s, s + n))) == 1:   #check horizontal
                return True, player   

        for s in range(max(m-(n-1)*width, w), min(m, max_len - n*width + w) + width, width):   
            if len(set(states.get(i, -1) for i in range(s, s + n * width, width))) == 1:  #check vertical
                return True, player
        
        ps = [m-i*(width+1) for i in range(n)]   #positive slope
        for j in range(n):
            if (height-n >= h-j >=0) and (width-n >= w-j >=0):
                if len(set(states.get(i, -1) for i in range(ps[j], ps[j] + n*(width+1), width+1)))==1:  
                    return True, player
        
        ns = [m-i*(width-1) for i in range(n)]   #negative slope
        for j in range(n):
            if (height-n >= h-j >=0) and (width > w+j >= n-1):   #width opposite direction & sign   #note: no equal sign for width > w+j
                if len(set(states.get(i, -1) for i in range(ns[j], ns[j] + n*(width-1), width-1)))==1: 
                    return True, player

        return False, -1

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

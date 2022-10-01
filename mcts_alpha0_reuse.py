# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes
Adapted from https://github.com/junxiaosong/AlphaZero_Gomoku

Modified by Chi-Hang Suen:
# mcts_alpha0_resuse: based on version _explore, reuse search sub-tree for evaluation mode as well
# mcts_alpha0_explore: in get_action, also add back Dirichlet noise after MCTS search 
#            (in addition to Dirichlet noise inside MCTS search), add such explore config
# mcts_alpha0_refill: expand and refill with noise for root if self-play; 
#                    rename input state as board for _playout & get_move_probs to align terminology in game module
# mcts_alpha0: remove unnecessary steps in expand & by moving network evaluation to after checking end game or not
Fix bug in Dirichlet noise implementation in: MCTS_player & MCTS __init__, TreeNode expand, MCTS _playout, MCTS_player get_action.
Align with paper, set temperature as 1 for early moves during self-play: MCTS_player get_action
"""

import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            self._children[action] = TreeNode(self, prob)

    def expand_refill(self, action_priors):
        """Expand and/or refill new probs (due to Dirichlet noise at root) """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
            else:
                self._children[action]._P = prob

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000, is_selfplay=0, noise=0):    
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._is_selfplay = is_selfplay
        self.noise = noise        

    def _playout(self, board):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        board state is modified in-place, so a copy must be provided. 
        assume noise=0 """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            board.do_move(action)

        # Check for end of game.
        end, winner = board.game_end()
        if not end:
            #noise = self.noise  if node.is_root() and self._is_selfplay  else 0
            #get probs & evaluate leaf by network output
            action_probs, leaf_value = self._policy(board, noise=0)
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (1.0 if winner == board.get_current_player() else -1.0)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, board, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        board state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        if self._is_selfplay:
            action_probs, leaf_value = self._policy(board, self.noise)  #Dirichlet noise at root
            self._root.expand_refill(action_probs)
            self._root.update_recursive(-leaf_value)
        for n in range(self._n_playout-1):
            state_copy = copy.deepcopy(board)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0, noise=0, explore=0.25):      
        self.mcts = MCTS(policy_value_function, c_puct, n_playout, is_selfplay, noise)
        self._is_selfplay = is_selfplay
        self.e = explore

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):                                 
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            if board.last_move == -1: 
                self.reset_player()
            if not self._is_selfplay:    #update for opponent move
                self.mcts.update_with_move(board.last_move)  
            acts, probs = self.mcts.get_move_probs(board, temp)   #best move if temp near 0; 1 means proportional
            move_probs[list(acts)] = probs
            if self._is_selfplay and self.e:
                probs = (1-self.e)*probs + self.e * np.random.dirichlet(0.3*np.ones(len(probs)))
            move = np.random.choice(acts, p=probs)
            # update the root node and reuse the search tree
            self.mcts.update_with_move(move)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)

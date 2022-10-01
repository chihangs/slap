# -*- coding: utf-8 -*-
"""An implementation of the policyValueNet in PyTorch
Adapted from https://github.com/junxiaosong/AlphaZero_Gomoku

Modified by Chi-Hang Suen:
# Add slap to policy_value_fn
# add np.array() before convert to tensor to avoid being slow warning
Use torch.tanh instead as nn.functional.tanh is deprecated. 
""" 
# policy10a: allow model be loaded by CPU
# policy10: add use_slap usage: slap as supplement instead of replacement of data augmentation
# rename self.board_heoght to net_height for flexibility and readability
# policy9b: fix policy_value_fn: reshape act_probs before unslap!!!
# policy9a: use autoclip instead of absolute clip value
# policy9: import slap6 for cc_pos, stone_pos; clip gradients at (-1, 1); 
#          add option to normalize or not the probs before MCTS
# policy8: add self.in_channel, import slap4
#          process crop & centre (cc) info in policy_value_fn
# policy7: config SGD changed to optim: add AdamW option; 
#          calculate validation loss by allowing train_mode False
# policy6b: fix bug on net for extra_act_fc
# policy6: add dropout option & extra_act_fc option (extra FC layer for action)
# policy5: add SGD optimizer option
# policy4: policy_value_fn: add evaluation mode, especially needed due to batchnorm, 
#          and with torch no grad which can also increase inference speed,
#          noise applied after (instead of before) masking out illegal actions,
#          remove noise count & sum
# policy3: output value_loss, policy_loss, refactoring noise code line
# policy2: let noise vary against leaf value, i.e. underdog plays more randomly
#          add self.noise.count, self.noise.sum to pass to main program
# from policy_slap_noise_v7b, save computation by flatenning later; 
#     combine slap and non-slap, allow different networks & config Dirichlet, L2, slap opening
# v7b: remove upside down as current state is no longer stored upside down; 
#       make policy_value_fn output format consistent 
# v5.6 allow config noise
# v5.4 fix rot90 axes & and flipud (use axis instead) & dirichlet in policy_value_net_fn; 
#     Fix log_softmax deprecation warning (use dim=-1, which gives same values as before)
# v5.1 fix bug in slap usage 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from slap6 import slap, unslap, cc_state, cc_pos, stone_pos
from autoclip.torch import QuantileClip


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, net_height, dropout=0, extra_act_fc=0, in_channel=4):
        super(Net, self).__init__()
        self.board_width = board_width
        self.net_height = net_height
        self.dropout = nn.Dropout(dropout)
        self.extra_act_fc = extra_act_fc
        # common layers
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*net_height, board_width*net_height)
        self.act_fc2 = nn.Linear(board_width*net_height, board_width*net_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*net_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = self.dropout(x_act.view(-1, 4*self.board_width*self.net_height))
        x_act = self.act_fc1(x_act)
        if self.extra_act_fc:
            x_act = self.act_fc2(self.dropout(F.relu(x_act)))
        x_act = F.log_softmax(x_act, dim=-1)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = self.dropout(x_val.view(-1, 2*self.board_width*self.net_height))
        x_val = self.dropout(F.relu(self.val_fc1(x_val)))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val

class ResidualBlock(nn.Module):
    #Source: https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/
    def __init__(self, in_channels, out_channels, stride=[1, 1], downsample=None):
        """ A basic residual block of ResNet, allow downsampling of features.
            in_channels: Number of channels that the input have
            out_channels: Number of channels that the output have
            stride: strides in convolutional layers
            downsample: A callable to be applied before addition of residual mapping """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        # applying a downsample function before adding it to the output
        if(self.downsample is not None):
            residual = self.downsample(residual)
        out = F.relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        # note that adding residual before activation 
        out = out + residual
        out = F.relu(out)
        return out    
    
class Res_Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, net_height, num_ResBlock=3, dropout=0, extra_act_fc=0, in_channel=4):
        super(Res_Net, self).__init__()
        self.board_width = board_width
        self.net_height = net_height
        self.num_ResBlock = num_ResBlock
        self.dropout = nn.Dropout(dropout)
        self.extra_act_fc = extra_act_fc
        # common layers
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.ResBlock = ResidualBlock(256,256)
        # action policy layers
        self.act_conv1 = nn.Conv2d(256, 2, kernel_size=1)
        self.act_bn = nn.BatchNorm2d(2)
        self.act_fc1 = nn.Linear(2*board_width*net_height, board_width*net_height)
        self.act_fc2 = nn.Linear(board_width*net_height, board_width*net_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(256, 1, kernel_size=1)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(board_width*net_height, 256)
        self.val_fc2 = nn.Linear(256, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.bn1(self.conv1(state_input)))
        for i in range(self.num_ResBlock):
            x = self.ResBlock(x)  #default only use 3 ResBlock, while paper used either 19 or 39 blocks
        # action policy layers
        x_act = F.relu(self.act_bn(self.act_conv1(x)))
        x_act = self.dropout(x_act.view(-1, 2*self.board_width*self.net_height))
        x_act = self.act_fc1(x_act)
        if self.extra_act_fc:
            x_act = self.act_fc2(self.dropout(F.relu(x_act)))
        x_act = F.log_softmax(x_act, dim=-1)
        # state value layers
        x_val = F.relu(self.val_bn(self.val_conv1(x)))
        x_val = self.dropout(x_val.view(-1, self.board_width*self.net_height))
        x_val = self.dropout(F.relu(self.val_fc1(x_val)))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val

class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, net_height, model_file=None, use_slap=False, num_ResBlock=0, 
                 L2=1e-4, opening=None, alpha=0.3, optimizer='Adam', dropout=0, extra_act_fc=0, in_channel=4, cc_fn=None, normalized=True, use_gpu=torch.cuda.is_available()):
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.board_width = board_width
        self.net_height = net_height
        self.use_slap = use_slap
        self.l2_const = L2  # coef of l2 penalty
        self.opening = opening
        self.alpha = alpha
        self.in_channel = in_channel
        self.cc_fn = cc_fn
        self.normalized = normalized
        # the policy value net module
        if num_ResBlock > 0:
            self.policy_value_net = Res_Net(board_width, net_height, num_ResBlock, dropout, extra_act_fc, in_channel).to(self.device)
        else:
            self.policy_value_net = Net(board_width, net_height, dropout, extra_act_fc, in_channel).to(self.device)
        # set optimizer and initial model parameters (if any), lr will be set in train_step
        if optimizer =='SGD':
            self.optimizer = optim.SGD(self.policy_value_net.parameters(), lr=0, momentum=0.9, weight_decay=self.l2_const) 
        elif optimizer == 'Adam': 
            self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const) #pytorch default weight decay 0
        elif optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.policy_value_net.parameters(), weight_decay=self.l2_const) #pytorch default weight decay 0.01
        self.optimizer = QuantileClip.as_optimizer(optimizer=self.optimizer, quantile=0.9, history_length=1000)
        if model_file:
            net_params = torch.load(model_file, map_location=self.device)
            self.policy_value_net.load_state_dict(net_params)
            self.policy_value_net.to(self.device)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board, noise=0):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        #list of legal moves in integers 0, 1, 2...; slap opening only used for first move if given
        legal_positions = board.availables if (not self.opening) or board.last_move > -1  else self.opening
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, board.height, board.width))
        if self.use_slap == 'add':
            slap_state, temp_flip, temp_i = slap(current_state)
            current_state = np.concatenate((current_state, np.zeros((1,4,1,board.width)), slap_state), axis=-2)
        if self.use_slap == 'replace':  
            current_state, temp_flip, temp_i = slap(current_state)
        if self.cc_fn:  #i.e. crop & centre info is required; not used with 'add' of use_slap 
            current_state = np.expand_dims(np.concatenate((current_state[0], self.cc_fn(current_state[0]))), axis=0)
        self.policy_value_net.eval()   #evaluation mode needed for batchnorm
        with torch.no_grad():          #faster for inference
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state.copy())).to(self.device).float())
        self.policy_value_net.train()
        # format output from network
        value = value.to(torch.device('cpu')).data[0][0].numpy()
        act_probs = np.exp(log_act_probs.data.to(torch.device('cpu')).numpy())
        if self.use_slap == 'add':
            act_probs = act_probs.reshape(self.net_height, self.board_width)
            act_probs_slap = act_probs[board.height+1:self.net_height]   #skip middle padding row
            act_probs_extra = unslap(act_probs_slap, temp_flip, temp_i)  #reverse slap
            act_probs = act_probs[0:board.height] + act_probs[board.height+1:self.net_height]  #sum instead of mean due to log_softmax
        if self.use_slap == 'replace': 
            act_probs = unslap(act_probs.reshape(self.net_height, self.board_width), temp_flip, temp_i)  #reshape & reverse slap for act_probs
        # move representation format of board 
        act_probs = act_probs.flatten()    
        probs_legal = act_probs[legal_positions]
        if self.normalized:
            probs_legal /= probs_legal.sum()
        if noise:       #weight of randomness slides over (noise[0], noise[1]) against leaf value
            if noise[0]>0:
                weight = (noise[0] + noise[1] + (noise[0]-noise[1])*value)/2   
                probs_legal = (1-weight) * probs_legal + weight * np.random.dirichlet(self.alpha*np.ones(len(probs_legal)))
        return zip(legal_positions, probs_legal), value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr, train_mode=True):
        """perform a training step"""
        # wrap in Variable
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float, requires_grad=train_mode).to(self.device)
        mcts_probs = torch.tensor(np.array(mcts_probs), dtype=torch.float, requires_grad=train_mode).to(self.device)
        winner_batch = torch.tensor(winner_batch, dtype=torch.float, requires_grad=train_mode).to(self.device)
        if train_mode:
            self.policy_value_net.train()
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # set learning rate
            set_learning_rate(self.optimizer, lr)
        else:
            self.policy_value_net.eval()
        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        if train_mode:
            # backward and optimize
            loss.backward()
            self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs)*log_act_probs, 1))
        return loss.item(), entropy.item(), value_loss.item(), policy_loss.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)



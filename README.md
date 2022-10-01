## Switchable Lightweight Anti-symmetric Processing (SLAP) with CNN - Application in Gomoku Reinforcement Learning

### Abstract
I created a novel method called SLAP to speed up convergence of machine learning. SLAP is a model-independent protocol to produce the same output given different transformation variants. It can be used upon any function or model to produce outputs that are invariant with regard to specified symmetric properties of the inputs. It can be viewed as standardization of symmetry, as opposed to standardization of scale. In a preliminary stage with experiment by synthetic states, CNN with SLAP had smaller validation losses than baseline (CNN without SLAP) for 77% of groups of hyperparameter and architecture combinations, despite that baseline was trained using 8 times the number of training samples by data augmentation. Among selected models from this preliminary stage and upon further training, it was found that SLAP improved the convergence speed of neural network learning by 83% compared with its baseline control. In reinforcement learning for a board game called Gomoku (also called Five-in-a-Row or Gobang), AlphaGo Zero/AlphaZero algorithm with data augmentation was used as the baseline, and the use of SLAP reduced the number of training samples by a factor of 8 and achieved similar winning rate against the same evaluator.

### Usage
Original usage was to test the new method SLAP in neural network learning and reinforcement learning. Users may also experiment without the new method SLAP, as there are options to test various hyperparameters and network architectures by just changing configurations in the file(s).

### Deep Learning Framework
Pytorch

### Install
```
pip install autoclip  
```

### Quick Start
To play with AI, run the following script from the directory:  
```
python human_play_v12.py  
```
To train the AI model from scratch, set your configuarions inside the file and run:   
```
python train_v12.py
```

### Overview of Files

#### New Method File
slap6b.py: the most updated file for the new method SLAP, with relevant functions. Used by training files and experiment files.

#### Training Files
The 6 files below for training AI were substantially upgraded from https://github.com/junxiaosong/AlphaZero_Gomoku which had copyright © 2017 Junxiao Song and MIT license. My upgrades include multi-processing and multi-tiers of evaluations, fixing minor bugs, aligining with AlphaZero paper more closely, adaptively decreasing learning rate based on standard deviation of validation loss, speeding up by 100% by consistent data format and quicker winner check, export of loss and result data, aligning game positions to array index style (instead of co-ordinate style), more options to set configurations e.g. options to use n layers of residual blocks and extra FC layer for action etc. See details of modifications in code comments and report.

game_array3.py, mcts_pure.py, mcts_alpha0_reuse.py, policy10a.py, train_v12.py, human_play_v12.py

#### Experiment Files
synthetic.py: create synthetic Gomoku states for testing neural network learning (i.e. decouple from dynamics of reinforcement learning)
validation_loss4a.py: use synthetic.py to calculate validation and training losses for only neural network learning in different experiments
train_multi2.py: train AI with multiple settings for experiments

#### Outdated Files
Marked as 'outdate', these files are kept as they were used by experiment files at that time and kept as is. The outdated version did not affect materially (if any) the experiment files for the purpose at that time. If users are using experiment files and woud like to change some configurations or features, they are advised to read about changes among different versions to decide whether to change to latest version for both validity and compatibility, based on their usage. 

### Documentation of the Experiments
See report.


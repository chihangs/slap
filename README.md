## Switchable Lightweight Anti-symmetric Processing (SLAP) with CNN - Application in Gomoku Reinforcement Learning

### Published Paper
Suen, C. H. & Alonso, E. (2023). Switchable Lightweight Anti-Symmetric Processing (SLAP) with CNN Outspeeds Data Augmentation by Smaller Sample – Application in Gomoku Reinforcement Learning. In Berndt Müller (Ed.), Proceedings of AISB Convention 2023 (pp. 69-75). AISB.
https://aisb.org.uk/wp-content/uploads/2023/05/aisb2023.pdf

### Abstract
To replace data augmentation, this paper proposed a method called SLAP to intensify experience to speed up machine learning and reduce the sample size. SLAP is a model-independent protocol/function to produce the same output given different transformation variants. SLAP improved the convergence speed of convolutional neural network learning by 83% in the experiments with Gomoku game states, with only one eighth of the sample size compared with data augmentation. In reinforcement learning for Gomoku, using AlphaGo Zero/AlphaZero algorithm with data augmentation as baseline, SLAP reduced the number of training samples by a factor of 8 and achieved similar winning rate against the same evaluator, but it was not yet evident that it could speed up reinforcement learning. The benefits should at least apply to domains that are invariant to symmetry or certain transformations. As future work, SLAP may aid more explainable learning and transfer learning for domains that are not invariant to symmetry, as a small step towards artificial general intelligence.

Keywords — data augmentation, convolutional neural network, symmetry invariant, group transformation, data preprocessing, SLAP, reinforcement learning


### Usage
Original usage was to test the new method SLAP in neural network learning and reinforcement learning about Gomoku. Users may also experiment without the new method SLAP, as there are options to test various hyperparameters and network architectures by just changing configurations in the file(s).

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
To train the AI model from scratch, set your configurations inside the file and run:   
```
python train_v12.py
```

### Overview of Files

#### New Method File
slap6b.py: the most updated file for the new method SLAP, with relevant functions. Used by training files and experiment files.

#### Training Files
game_array3.py, mcts_pure.py, mcts_alpha0_reuse.py, policy10a.py, train_v12.py, human_play_v12.py

The 6 files above for training AI were upgraded from https://github.com/junxiaosong/AlphaZero_Gomoku which had copyright © 2017 Junxiao Song and MIT license. My upgrades include multi-processing and multi-tiers of evaluations, fixing minor bugs, aligining with AlphaZero paper more closely, adaptively decreasing learning rate based on standard deviation of validation loss, speeding up by 100% by consistent data format and quicker winner check, export of loss and result data, aligning game positions to array index style (instead of co-ordinate style), more options to set configurations e.g. options to use n layers of residual blocks and extra FC layer for action etc. See details of modifications in code comments and report.

#### Experiment Files
synthetic.py: create synthetic Gomoku states for testing neural network learning (i.e. decouple from dynamics of reinforcement learning)
validation_loss4a.py: use synthetic.py to calculate validation and training losses for only neural network learning in different experiments
train_multi2.py: train AI with multiple settings for experiments
AIvsAI_flexible.py: AI playing with another AI model; greedy and deterministic - exact repetition of games for same start player as noise is always disabled for non self-play. If time allows, build the noise enable option.

#### Outdated Files
Marked as 'outdate', these files are kept as they were used by experiment files at that time and kept as is. The outdated version did not affect materially (if any) the experiment files for the purpose at that time. If users are using experiment files and woud like to change some configurations or features, they are advised to read about changes among different versions to decide whether to change to latest version by taking into account the compatibility and the validity based on their usage. 

### Documentation of the Experiments
See report.


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

### Files
#### Reinforcement Learning


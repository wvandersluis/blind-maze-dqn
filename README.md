# blind-maze-dqn

Training of a Deep Neural Network to find its way through a maze using Deep Q-Learning.


 - 'MazeGenerationKeyInput.py' allows for a visualization of what the ANN 'sees' when solving the maze.

 - 'mazeDQN.py' is the implementation of the standard DQN (including dropout layers) as described in DeepMind's 2015 paper.

 - 'mazeDoubleDQN.py' is the implementation of the Double DQN as described by Hasselt et al. (2016) extended to use soft target updates.

 - 'evalDQN.py' can be run to visually evaluate a DQN running through the maze. It is set to run the best obtained network, the state dictionary of which is stored in 'example_model.pth'
 - 

Enjoy!

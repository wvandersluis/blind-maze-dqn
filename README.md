# blind-maze-dqn

_Training of a Deep Neural Network to find its way through a maze using Deep Q-Learning._

 - 'MazeGenerationKeyInput.py' allows for an interactive visualization of what the ANN 'sees' when solving the maze.

 - 'mazeDQN.py' is the implementation of the standard DQN (with dropout layers added) as described in DeepMind's 2015 paper.

 - 'mazeDoubleDQN.py' is the implementation of the Double DQN as described by Hasselt et al. (2016) extended to use soft target updates.

 - 'evalDQN.py' can be run to visually evaluate a DQN running through the maze. It is set to run the best obtained network. The state dictionary of this network is stored in 'example_model.pth'

Enjoy!

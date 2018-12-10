
Thie repo uses Q-learning to train an epsilon-greedy agent to find the shortest path between position `random start to random goal positions of a 2D rectangular grid in the 2D GridWorld environment.

The agent is restricted to move itself up/down/left/right by 1 grid square per action. The agent receives a `0` penalty for every action not reaching the terminal state and a `1` reward upon reaching the terminal state.

The agent exploration parameter `epsilon` also decays by a multiplicative constant after every training episode. 




Thie repo uses Q-learning to train an epsilon-greedy agent to find the shortest path between a given start and goal postions  in the 2D GridWorld environment.

The agent is restricted to move itself up/down/left/right by 1 grid square per action. The agent receives a reward of 1 for reaching the goal state. Otherwise, it get a 0 reward.

The agent exploration parameter `epsilon` also decays by a multiplicative constant after every training episode. 



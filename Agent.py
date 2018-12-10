import os, sys, random, operator
import numpy as np
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, env, nRow, nCol):
        # Store state and action dimension

        # Agent learning parameters
        self.epsilon = 1.0  # initial exploration probability
        self.epsilon_decay = 0.99  # epsilon decay after each episode
        self.learning_rate = 0.99  # learning rate
        self.discount_factor = 0.99  # reward discount factor
        self.Q=np.zeros([nRow*nCol,4])


    def get_action(self, env,nRow,nCol):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(env.allowed_actions())
        else:

        # exploit on allowed actions
            state = env.state;
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0]*nRow+state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def train(self, memory):

        (state, action, state_next, reward, done) = memory
        print("Memory",memory)
        self.Q[state,action]+=self.learning_rate * \
                              (reward + self.discount_factor * np.max(self.Q[state_next,:]) - self.Q[state,action])

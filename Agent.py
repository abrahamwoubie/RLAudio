import os, sys, random, operator
import numpy as np
import matplotlib.pyplot as plt
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Agent:

    def __init__(self, env, nRow, nCol):
        # Store state and action dimension

        # Agent learning parameters
        # self.epsilon = 1.0  # initial exploration probability
        # self.epsilon_decay = 0.99  # epsilon decay after each episode
        # self.learning_rate = 0.99  # learning rate
        # self.discount_factor = 0.99  # reward discount factor
        self.Q=np.zeros([nRow*nCol,4])


        self.state_size = 16
        self.action_size = 4
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()



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


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=2, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
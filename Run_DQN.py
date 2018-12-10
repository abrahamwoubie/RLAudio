import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from Agent import Agent
from Environment import Environment

N_episodes = 10
reward_List=[]
number_of_iterations_per_episode=[]
number_of_episodes=[]

if __name__ == "__main__":
    env = Environment(8,8)
    agent = Agent(env, 8, 8)

    #state_size = 16
    #action_size = 4
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for episode in range(N_episodes):
        episode=episode+1
        state = env.reset()
        state = np.reshape(state, [1, 2])
        iteration=0
        number_of_episodes.append(episode)
        for iteration in range(100):
            # env.render()
            iteration+=1
            action = agent.get_action(env, 4, 4)  # get action
            next_state, reward, done = env.step(action)
            #reward = reward if not done else 0
            next_state = np.reshape(next_state, [1, 2])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            if len(agent.memory) > batch_size:
                 agent.replay(batch_size)
        print("[episode {}/{}] Number of Iterations = {}, Reward  = {}".format(
            episode, N_episodes, iteration, reward))
        number_of_iterations_per_episode.append(iteration)
        reward_List.append(reward)
percentage_of_successful_episodes=(sum(reward_List)/N_episodes)*100

print("Percentage of Successful Episodes is {} {}".format(percentage_of_successful_episodes,'%'))
fig = plt.figure()
fig.suptitle('Q-Learning', fontsize=12)
plt.plot(np.arange(len(number_of_episodes)), number_of_iterations_per_episode)
plt.ylabel('Number of Iterations')
plt.xlabel('Episode')
# plt.grid(True)
#plt.savefig("Q_Learning_10_10.png")
plt.show()
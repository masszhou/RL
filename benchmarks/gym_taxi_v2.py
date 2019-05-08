import gym
import numpy as np
import os, sys
import time
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# To find local version of the library
sys.path.append(ROOT_DIR)

from models.sarsa import SARSA
from models.sarsa import QLearning
from models.sarsa import ExpectedSARSA
from models.sarsa import interact

env = gym.make('Taxi-v2')
agent_sarsa = SARSA(env.action_space.n)
agent_maxsarsa = QLearning(env.action_space.n)
agent_expsarsa = ExpectedSARSA(env.action_space.n)

start = time.time()
agent_sarsa, avg_rewards_sarsa = interact(env, agent_sarsa, max_steps_episode=300, num_episodes=5000)
print("\nSARSA total time = ", time.time() - start)
start = time.time()
agent_maxsarsa, avg_rewards_maxsarsa = interact(env, agent_maxsarsa, max_steps_episode=300, num_episodes=5000)
print("\nQLearning total time = ", time.time() - start)
start = time.time()
agent_expsarsa, avg_rewards_exppsarsa = interact(env, agent_expsarsa, max_steps_episode=300, num_episodes=5000)
print("\nExpected SARSA total time = ", time.time() - start)

plt.plot(np.array(avg_rewards_sarsa), c="r")
plt.plot(np.array(avg_rewards_maxsarsa), c="g")
plt.plot(np.array(avg_rewards_exppsarsa), c="b")

plt.show()
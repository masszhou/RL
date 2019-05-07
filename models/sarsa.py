import numpy as np
from collections import defaultdict
from collections import deque
import sys


class SARSA():
    def __init__(self,
                 nA,
                 gamma=0.9,
                 alpha=0.2,
                 epsilon_start=1.0,
                 epsilon_decay=0.9999,
                 epsilon_min=0.001,
                 epsilon_fn=None):
        self.nA = nA
        self.action_space = list(range(self.nA))

        self.gamma = gamma

        self.alpha = alpha

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_epsilon = epsilon_fn if epsilon_fn is not None else self.decay_epislon

        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def update(self, sarsa_sample):
        # update with Temporal-Difference instead of complete episode
        # trick action id = action value, so we can use self.Q[s][a] for q(s,a)
        s, a, r, s_next, a_next = sarsa_sample[0:5]
        old_Q = self.Q[s][a]
        self.Q[s][a] = old_Q + self.alpha * (r + self.gamma * self.Q[s_next][a_next] - old_Q)
        self.update_epsilon()

    def choose_action(self, state):
        prob_choose = np.ones(self.nA) / self.nA * self.epsilon
        action_value = self.Q[state]
        prob_choose[np.argmax(action_value)] = 1 - (self.nA - 1.0)/self.nA * self.epsilon
        action = np.random.choice(self.action_space, p=prob_choose)
        return action

    def get_policy(self):
        return dict((k, np.argmax(v)) for k, v in self.Q.items())

    def get_Qtable(self):
        return self.Q

    def reset_epislon(self):
        self.epsilon = self.epsilon_start

    def decay_epislon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


class MaxSARSA(SARSA):
    def update(self, sarsa_sample):
        s, a, r, s_next = sarsa_sample[0:4]  # do not need action_next
        # update with Temporal-Difference instead of complete episode
        # trick action id = action value, so we can use self.Q[s][a] for q(s,a)
        old_Q = self.Q[s][a]
        self.Q[s][a] = old_Q + self.alpha * (r + self.gamma * np.max(self.Q[s_next]) - old_Q)
        self.update_epsilon()


class QLearning(SARSA):
    def update(self, sarsa_sample):
        s, a, r, s_next = sarsa_sample[0:4]  # do not need action_next
        # update with Temporal-Difference instead of complete episode
        # trick action id = action value, so we can use self.Q[s][a] for q(s,a)
        old_Q = self.Q[s][a]
        self.Q[s][a] = old_Q + self.alpha * (r + self.gamma * np.max(self.Q[s_next]) - old_Q)
        self.update_epsilon()


class ExpectedSARSA(SARSA):
    def update(self, sarsa_sample):
        s, a, r, s_next = sarsa_sample[0:4]  # do not need action_next
        # update with Temporal-Difference instead of complete episode
        # trick action id = action value, so we can use self.Q[s][a] for q(s,a)
        old_Q = self.Q[s][a]

        prob = np.ones(self.nA) / self.nA * self.epsilon
        action_value = self.Q[s_next]
        prob[np.argmax(action_value)] = 1 - (self.nA - 1.0) / self.nA * self.epsilon

        self.Q[s][a] = old_Q + self.alpha * (r + self.gamma * np.dot(self.Q[s_next], prob) - old_Q)
        self.update_epsilon()


def interact(env, agent, max_steps_episode=300, num_episodes=5000, window=100):
    avg_rewards = deque(maxlen=num_episodes)  # measure performance

    window_rewards = deque(maxlen=window)  # when full, oldest item will automatically pop up

    for i_episode in range(num_episodes):
        # reset env for each episode
        state = env.reset()
        action = None
        # update epsilon external
        # agent.epsilon = 1.0 / (i_episode + 1)

        step_counter = 0
        sum_reward = 0

        while True:
            if action is None:
                action = agent.choose_action(state)
            state_next, reward, done, info = env.step(action)
            action_next = agent.choose_action(state_next)
            sarsa_sample = [state, action, reward, state_next, action_next]

            agent.update(sarsa_sample)

            action = action_next
            state = state_next

            sum_reward += reward
            step_counter += 1

            if done is True:
                window_rewards.append(sum_reward)
                break
            elif step_counter > max_steps_episode:
                break

        if len(window_rewards) > 0:
            avg_rewards.append(np.mean(window_rewards))

        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, max(avg_rewards)), end="")
        sys.stdout.flush()

    return agent, avg_rewards


if __name__ == "__main__":
    import gym
    import time
    import matplotlib.pyplot as plt

    env = gym.make('CliffWalking-v0')

    agent_sarsa = SARSA(env.action_space.n)
    agent_maxsarsa = MaxSARSA(env.action_space.n)
    agent_expsarsa = ExpectedSARSA(env.action_space.n)

    start = time.time()
    agent_sarsa, avg_rewards_sarsa = interact(env, agent_sarsa, max_steps_episode=300, num_episodes=5000)
    print("\nSARSA total time = ", time.time()-start)
    start = time.time()
    agent_maxsarsa, avg_rewards_maxsarsa = interact(env, agent_maxsarsa, max_steps_episode=300, num_episodes=5000)
    print("\nMax SARSA total time = ", time.time()-start)
    start = time.time()
    agent_expsarsa, avg_rewards_exppsarsa = interact(env, agent_expsarsa, max_steps_episode=300, num_episodes=5000)
    print("\nExpected SARSA total time = ", time.time()-start)

    plt.plot(np.array(avg_rewards_sarsa), c="r")
    plt.plot(np.array(avg_rewards_maxsarsa), c="g")
    plt.plot(np.array(avg_rewards_exppsarsa), c="b")

    plt.show()
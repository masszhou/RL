import numpy as np
from collections import defaultdict
from collections import deque
import sys


class MCControl():
    """
    for episodic tasks, learn from each episode
    """
    def __init__(self):
        self.gamma = 1.0  # discount factor for rewards
        self.epsilon = 1.0  # greedy factor, 1->random, 0->optimal
        self.epsilon_start = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.05
        self.alpha = 0.02  # learning rate
        self.nA = 2  # action space size
        self.action_space = list(range(self.nA))  # action space
        self.Q = defaultdict(lambda: np.zeros(self.nA))  # Q table

    def update(self, episode):
        # episode, [[s,a,r],[s,a,r],...]
        # update Q table with accumulated rewards from a given episode

        states, actions, rewards = zip(*episode)
        discount = [self.gamma ** i for i in range(len(rewards))]
        discount = np.array(discount[::-1])
        discounted_rewards = np.array(rewards) * discount

        for i, state in enumerate(states):
            # sequence, index from start to end, i=0 is start, -1 is end
            # sum rewards from step i to the end
            old_Q = self.Q[state][actions[i]]
            self.Q[state][actions[i]] = old_Q + self.alpha * (np.sum(discounted_rewards[i::]) - old_Q)

    def choose_action(self, state):
        action_value = self.Q[state]
        prob_list = np.ones(self.nA) * self.epsilon / self.nA  # eps to choose others
        prob_list[np.argmax(action_value)] = 1 - (self.nA - 1) / self.nA * self.epsilon  # 1-eps to choose best
        action = np.random.choice(self.action_space, p=prob_list)
        return action

    def policy(self):
        return dict((k, np.argmax(v)) for k, v in self.Q.items())

    def Qtable(self):
        return self.Q

    def learn_from_episodes(self, generate_episode, n_episodes=50000):
        self.epsilon = self.epsilon_start
        for i_episode in range(n_episodes):
            print("\rEpisode {}/{}.".format(i_episode, n_episodes), end="")
            sys.stdout.flush()
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            episode = generate_episode(self.choose_action)
            self.update(episode)


def interact(env, agent, max_steps_episode=100, num_episodes=20000, smooth_window=10):
    avg_rewards = deque(maxlen=num_episodes)  # measure performance
    state = env.reset()
    for i_episode in range(num_episodes):
        episode = []
        step_counter = 0
        sum_reward = 0
        succ_scores = deque(maxlen=max_steps_episode)
        while True:
            action = agent.choose_action(state)
            state_next, reward, done, info = env.step(action)
            episode.append([state, action, reward])
            sum_reward += reward
            step_counter += 1

            state = state_next
            if done is True:
                succ_scores.append(np.mean(sum_reward))
                break
            elif step_counter > max_steps_episode:
                break

        if len(succ_scores) > 0:
            avg_rewards.append(succ_scores)

        agent.update(episode)

    return agent, avg_rewards


if __name__ == "__main__":
    import gym
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    env = gym.make('Blackjack-v0')
    agent = MCControl()
    agent, best_avg_reward = interact(env, agent, num_episodes=500000)


    def plot_blackjack_values(V):

        def get_Z(x, y, usable_ace):
            if (x, y, usable_ace) in V:
                return V[x, y, usable_ace]
            else:
                return 0

        def get_figure(usable_ace, ax):
            x_range = np.arange(11, 22)
            y_range = np.arange(1, 11)
            X, Y = np.meshgrid(x_range, y_range)

            Z = np.array([get_Z(x, y, usable_ace) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
            ax.set_xlabel('Player\'s Current Sum')
            ax.set_ylabel('Dealer\'s Showing Card')
            ax.set_zlabel('State Value')
            ax.view_init(ax.elev, -120)

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(211, projection='3d')
        ax.set_title('Usable Ace')
        get_figure(True, ax)
        ax = fig.add_subplot(212, projection='3d')
        ax.set_title('No Usable Ace')
        get_figure(False, ax)
        plt.show()


    def plot_policy(policy):

        def get_Z(x, y, usable_ace):
            if (x, y, usable_ace) in policy:
                return policy[x, y, usable_ace]
            else:
                return 1

        def get_figure(usable_ace, ax):
            x_range = np.arange(11, 22)
            y_range = np.arange(10, 0, -1)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.array([[get_Z(x, y, usable_ace) for x in x_range] for y in y_range])
            surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2', 2), vmin=0, vmax=1, extent=[10.5, 21.5, 0.5, 10.5])
            plt.xticks(x_range)
            plt.yticks(y_range)
            plt.gca().invert_yaxis()
            ax.set_xlabel('Player\'s Current Sum')
            ax.set_ylabel('Dealer\'s Showing Card')
            ax.grid(color='w', linestyle='-', linewidth=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)
            cbar.ax.set_yticklabels(['0 (STICK)', '1 (HIT)'])

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(121)
        ax.set_title('Usable Ace')
        get_figure(True, ax)
        ax = fig.add_subplot(122)
        ax.set_title('No Usable Ace')
        get_figure(False, ax)
        plt.show()


    V_to_plot = dict((k, (k[0] > 18) * (np.dot([0.8, 0.2], v)) + (k[0] <= 18) * (np.dot([0.2, 0.8], v))) \
                     for k, v in agent.Q.items())
    # plot the state-value function
    plot_blackjack_values(V_to_plot)
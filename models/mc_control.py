import numpy as np
from collections import defaultdict
from collections import deque
import sys
from tqdm import tqdm


class MCControl():
    """
    for episodic tasks, learn from each episode
    """
    def __init__(self, nA):
        # discount factor for rewards
        self.gamma = 1.0
        # greedy factor, 1->random, 0->optimal
        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.05
        # learning rate
        self.alpha = 0.02
        # action space size
        self.nA = nA
        # action space
        self.action_space = list(range(self.nA))
        # Q table
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def update(self, episode):
        """
        updates the action-value function estimate using the most recent episode
        """
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
        """
        choose action with epsilon greedy
        """
        action_value = self.Q[state]
        prob_list = np.ones(self.nA) * self.epsilon / self.nA  # eps to choose others
        prob_list[np.argmax(action_value)] = 1 - (self.nA - 1) / self.nA * self.epsilon  # 1-eps to choose best
        action = np.random.choice(self.action_space, p=prob_list)
        return action

    def get_policy(self):
        return dict((k, np.argmax(v)) for k, v in self.Q.items())

    def get_Qtable(self):
        return self.Q

    def reset_epislon(self):
        self.epsilon = self.epsilon_start

    def decay_epislon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


def interact(env, agent, max_steps_episode=500, num_episodes=20000):
    avg_rewards = deque(maxlen=num_episodes)  # measure performance
    pbar = tqdm(total=num_episodes)

    for i_episode in range(num_episodes):
        state = env.reset()

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
        agent.decay_epislon()

        pbar.set_description('episode %i' % i_episode)
        pbar.update()

    pbar.close()
    return agent, avg_rewards


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

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


    env = gym.make('Blackjack-v0')
    agent = MCControl(env.action_space.n)
    agent, avg_rewards = interact(env, agent, num_episodes=200000)
    plot_policy(agent.get_policy())
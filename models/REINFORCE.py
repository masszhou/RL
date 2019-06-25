import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, action_size)

    def forward(self, x):
        net = F.relu(self.fc1(x))
        net = F.softmax(self.fc2(net), dim=1)
        return net


class Reinforce:
    def __init__(self, state_size, action_size, lr=0.01):
        self.policy = Policy(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr)

    def act(self, state):
        """
        :param state:
        :return: scalar, torch.Tensor
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        param = self.policy.forward(state).cpu()

        # build a categorical distribution model, keep autograd chain
        m = torch.distributions.Categorical(param)  # m.probs does not equal to param
        # sample a action
        action = m.sample()
        return action.item(), m.log_prob(action)

    def learn(self, saved_log_prob, reward_list, gamma=1.0):
        """
        monte-carlo learning
        learn from one entire trajectory

        :param saved_log_prob: torch.Tensor, probability list of all actions in this trajectory
        :param reward_list: rewards in this trajectory
        :param gamma: scalar, discounted factor for sum of rewards
        :return: None
        """

        discounts = [gamma ** i for i in range(len(reward_list) + 1)]
        R = sum([a * b for a, b in zip(discounts, reward_list)])

        policy_loss = []
        for log_prob in saved_log_prob:
            policy_loss.append(-log_prob*R)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    import gym
    from collections import deque
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # or any '0,1'

    env = gym.make('CartPole-v0')
    env = env.unwrapped  # clean env, no attachment

    n_episodes = 1000
    max_t = 1000
    scores_log = []
    scores_deque = deque(maxlen=100)

    agent = Reinforce(4, 2)

    for i_episode in range(n_episodes):
        log_probs_list = []
        reward_list = []
        state = env.reset()

        # sample one trajectory
        for t in range(max_t):
            action, log_probs = agent.act(state)
            log_probs_list.append(log_probs)
            state, reward, done, info = env.step(action)
            reward_list.append(reward)
            if done is True:
                break
        scores_log.append(sum(reward_list))
        scores_deque.append(sum(reward_list))

        agent.learn(log_probs_list, reward_list)
        if i_episode % 100 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
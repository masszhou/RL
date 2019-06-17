import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from collections import deque
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()  # equivalent to nn.Module.__init__(self)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class DeepQNetwork:
    def __init__(self,
                 state_size,
                 action_size,
                 gamma=0.99,
                 learning_rate=0.0005,
                 update_every=100,
                 batch_size=64,
                 memory_size=int(1e5),
                 model_save_path="./models/",
                 log_save_path="./logs/",
                 output_graph=True):

        # super(DeepQNetwork, self).__init__()
        # ------------------------------------------
        # model parameters
        # ------------------------------------------
        self.action_size = action_size
        self.state_size = state_size
        self.lr = learning_rate
        self.gamma = gamma  # reward discounter
        self.update_every = update_every
        self.memory_size = memory_size
        self.batch_size = batch_size
        # initialize zero memory [s, a, r, s_, done]
        self.memory = deque(maxlen=self.memory_size)
        # total learning step
        self.learn_step_counter = 0

        # ------------------------------------------
        # model definition
        # ------------------------------------------
        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def act(self, state, eps):
        if np.random.uniform() < eps:
            action = np.random.randint(0, self.action_size)
        else:
            # forward feed the state and get q value for every actions
            with torch.no_grad():
                actions_value = self.policy_net(torch.from_numpy(state).float().unsqueeze(0))  # batch size = 1
            action = torch.argmax(actions_value).tolist()
        return action

    def step(self, s, a, r, next_s, done):
        # assert s,s_next -> np.array
        # assert a -> scalar
        # assert r -> scalar
        # assert done -> scalar
        self.memory.append(np.concatenate([s, [a], [r], next_s, [done]]))  # deque
        memory_size = len(self.memory)
        if memory_size < self.batch_size:
            return

        if self.learn_step_counter % self.update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        sample_index = np.random.choice(range(memory_size), size=self.batch_size)  # replace = False
        batch_memory = np.array([self.memory[i] for i in sample_index], dtype=np.float32)  # float32

        s_sample = batch_memory[:, 0:self.state_size]                              # shape=(#batch, n_features)
        a_sample = batch_memory[:, self.state_size:self.state_size+1]    # shape=(#batch,)
        r_sample = batch_memory[:, self.state_size+1:self.state_size+2]  # shape=(#batch,)
        s_next_sample = batch_memory[:, self.state_size+2:self.state_size*2+2]     # shape=(#batch, n_features)
        done_sample = batch_memory[:, self.state_size*2+2:self.state_size*2+3]     # shape=(#batch,)

        s_sample = torch.from_numpy(s_sample)
        a_sample = torch.from_numpy(a_sample)
        r_sample = torch.from_numpy(r_sample)
        s_next_sample = torch.from_numpy(s_next_sample)
        done_sample = torch.from_numpy(done_sample)

        # Q[s][a] = Q[s][a] + alpha * (r + gamma * np.max(Q[s_next]) - Q[s][a])
        # Q_target = r + gamma * np.max(target_net(next_s))
        # Q_policy = policy_net(s)[a]
        # find fix point of Q*=BQ*, thus min norm2(Q_target - Q_policy)

        Q_next = self.target_net(s_next_sample).detach()
        Q_target = r_sample + self.gamma * torch.max(Q_next, 1)[0].view(-1, 1)
        # Q_target = r_sample + self.gamma * torch.max(self.target_net(s_sample)) * (1 - done_sample)

        Q_policy = self.policy_net(s_sample).gather(1, a_sample.long())  # index must be Long Int

        loss = F.mse_loss(Q_target, Q_policy)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1


if __name__ == "__main__":
    import gym
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # or any '0,1'

    env = gym.make('CartPole-v0')
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Solved Requirements
    # Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = env.unwrapped

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    agent = DeepQNetwork(state_size=env.observation_space.shape[0],
                         action_size=env.action_space.n,
                         learning_rate=0.01)

    N_EPISODES = 200
    scores_log = []

    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    eps = eps_start

    for i_episode in range(N_EPISODES):

        state = env.reset()
        steps = 0
        while True:
            env.render()
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            # the smaller theta and closer to center the better
            x, x_dot, theta, theta_dot = next_state
            # smaller reward, when not in the central
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # smaller reward, when not perpendicular
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            # # if not use customized reward, the last reward +1 with done==True should be considered
            # # as an end game action, like a penalty.
            # reward = -1*reward if done is True else reward

            agent.step(state, action, reward, next_state, done)

            steps += 1
            state = next_state
            eps = max(eps_end, eps_decay * eps)
            if done:
                break
            if steps > 500:
                # sucessful episode
                break

        scores_log.append(steps)
        print("----------")
        print("episode {}, epsilon {}, learn_steps {}, steps {}".format(i_episode,
                                                                        round(eps, 2),
                                                                        agent.learn_step_counter,
                                                                        steps))
    env.close()
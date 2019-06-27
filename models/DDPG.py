# learned from udacity drlnd and Grokking Deep Reinforcement Learning
import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import torchsummary

from networks import Actor
from networks import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG:
    def __init__(self,
                 state_size,
                 action_size,
                 memory_size=int(1e5), # replay buffer size
                 batch_size=128,       # minibatch size
                 gamma=0.99,           # discount factor
                 tau = 1e-3,           # for soft update of target parameters
                 lr_actor=1e-4,
                 lr_critic=3e-4,
                 random_seed=2):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.params = {"lr_actor": lr_actor,
                       "lr_critic": lr_critic,
                       "gamma": gamma,
                       "tau": tau,
                       "memory_size": memory_size,
                       "batch_size": batch_size,
                       "optimizer": "adam"}

        self.actor_local = Actor(state_size, action_size, seed=random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed=random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        self.critic_local = Critic(state_size, action_size, seed=random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed=random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic)

        self.memory = ReplayBuffer(action_size, memory_size, batch_size, random_seed)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.params["batch_size"]:
            experiences = self.memory.sample()
            self.learn(experiences, self.params["gamma"])

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # ------------------------------------------
        # update critic
        # ------------------------------------------
        # recall DQN
        # Q[s][a] = Q[s][a] + alpha * (r + gamma * np.max(Q[s_next]) - Q[s][a])
        # thus, here
        # Q_local = Q[s][a]
        #         = critic_local(s, a)
        # Q_target = r + gamma * np.max(Q[s_next])
        #          = r + gamma * (critic_target[s_next， actor_target(s_next)])
        #
        # calculate np.max(Q[s_next]) with critic_target[s_next， actor_target(s_next)]
        # because actor suppose to output action which max Q(s)
        #
        # loss = mse(Q_local - Q_target)

        best_actions = self.actor_target(next_states)  # supposed to be best actions, however
        Q_next_max = self.critic_target(next_states, best_actions)
        Q_target = rewards + gamma * Q_next_max * (1 - dones)
        # Q_target_detached = Q_target.detach()

        Q_local = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q_local, Q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------------------------
        # update critic
        # ------------------------------------------
        # suppose critic(s,a) give us q_max as a baseline or guidance
        # we want actor(s) to output the right a
        # which let critic(s,a)->q_max happen
        # so we want find a_actor to max Q_critic(s, a)
        # a_actor is function of θ
        # so the gradient is dQ/da*da/dθ
        actions_pred = self.actor_local(states)
        Q_baseline = self.critic_local(states, actions_pred)
        actor_loss = -Q_baseline.mean()  # I think this is a good trick to make loss to scalar

        # note, gradients from both actor_local and critic_local will be calculated
        # however we only update actor_local
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.params["tau"])
        self.soft_update(self.actor_local, self.actor_target, self.params["tau"])

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt

    env = gym.make('Pendulum-v0')
    env.seed(2)

    agent = DDPG(state_size=3, action_size=1)


    def train_ddpg(n_episodes=1000, max_t=300, print_every=100):
        scores_deque = deque(maxlen=print_every)
        scores = []
        for i_episode in range(1, n_episodes + 1):
            state = env.reset()
            agent.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_deque.append(score)
            scores.append(score)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            if i_episode % print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        return scores


    scores = train_ddpg()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
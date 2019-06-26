# learned from udacity drlnd and Grokking Deep Reinforcement Learning
import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary as torch_summary

from policy_factory import Actor
from policy_factory import Critic

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class DDPG:
    def __init__(self,
                 state_size,
                 action_size,
                 memory_size=int(1e5), # replay buffer size
                 batch_size=128,       # minibatch size
                 gamma=0.99,           # discount factor
                 tau = 1e-3,           # for soft update of target parameters
                 lr_actor=0.0001,
                 lr_critic=0.001,
                 random_seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.params = {"lr_actor": lr_actor,
                       "lr_critic": lr_critic,
                       "gamma": gamma,
                       "tau": tau,
                       "memory_size": memory_size,
                       "batch_size": batch_size,
                       "optimizer": "adam"}

        self.actor_local = Actor(state_size, action_size)
        self.actor_target = Actor(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        self.critic_local = Critic(state_size, action_size)
        self.critic_target = Critic(state_size, action_size)
        self.critic_optimizer = optim.Adam(self.critic_target.parameters(), lr=lr_critic)

        self.memory = ReplayBuffer(action_size, memory_size, batch_size, random_seed)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.params["batch_size"]:
            experiences = self.memory.sample()
            self.learn(experiences, self.params["gamma"])

    def learn(self, experiences, gamma):
        pass

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
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


if __name__ == "__main__":
    import gym
    env = gym.make('Pendulum-v0')

    agent = DDPG(state_size=3, action_size=1)
    agent.summary()
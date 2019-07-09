import numpy as np
import torch


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    explore around mu
    """
    def __init__(self, action_dimension, scale=1, mu=0, theta=0.15, sigma=0.5):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.action_dimension)
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()


# https://github.com/gtg162y/DRLND/blob/master/P3_Collab_Compete/Tennis_Udacity_Workspace.ipynb
class Gaussian:
    def __init__(self, action_dimension, scale=0.1, mu=0, sigma=0.5):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.sigma = sigma

    def noise(self):
        noise = self.sigma * np.random.randn(self.action_dimension)
        return torch.from_numpy(noise).float()

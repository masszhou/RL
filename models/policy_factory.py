# learned from udacity DRLND showcase codes
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """
    input: state
    output: action
    """
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        #
        # init.xavier_uniform_(self.fc1.weight)
        # init.xavier_uniform_(self.fc2.weight)
        # init.xavier_uniform_(self.fc3.weight)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        net = F.relu(self.fc1(state))
        net = F.relu(self.fc2(net))
        net = F.tanh(self.fc3(net))
        return net


class Critic(nn.Module):
    """
    input: state, action
    output: Q_value
    """
    # def __init__(self, state_size, action_size, fc1_units=400, fc2_units=300):
    #     super(Critic, self).__init__()
    #     self.fc1 = nn.Linear(state_size + action_size, fc1_units)
    #     self.fc2 = nn.Linear(fc1_units, fc2_units)
    #     self.fc3 = nn.Linear(fc2_units, 1)
    #
    #     init.xavier_uniform_(self.fc1.weight)
    #     init.xavier_uniform_(self.fc2.weight)
    #     init.xavier_uniform_(self.fc3.weight)
    #
    # def forward(self, state, action):
    #     s_a = torch.cat([state, action], dim=1)
    #     net = F.relu(self.fc1(s_a))
    #     net = F.relu(self.fc2(net))
    #     net = self.fc3(net)  # no activation ?
    #     return net

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
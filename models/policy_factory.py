# learned from udacity DRLND showcase codes
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init


class Actor(nn.Module):
    """
    input: state
    output: action
    """
    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=128, seed=42):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)

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
    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=128, seed=42):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)

    def forward(self, state, action):
        s_a = torch.cat([state, action], dim=1)
        net = F.relu(self.fc1(s_a))
        net = F.relu(self.fc2(net))
        net = self.fc3(net)  # no activation ?
        return net
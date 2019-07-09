import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # (4, 96, 96) take 4 frames as 1 state to infer speed information
        self.cnn_bass = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2),   # (8, 47, 47)
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (32, 11, 11)


        )
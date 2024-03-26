from .block import *
import torch.nn as nn
# define architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5),
            Block(64,64),
            Block(64,64),
            Block(64,64),
            Block(64,64),
            Block(64,64)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,5),
        )
    def forward(self, x):
        x = self.conv_layer(x)
        x = nn.Flatten()(x)
        x = self.fc_layer(x)
        return x


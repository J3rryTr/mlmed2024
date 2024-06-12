import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv0 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=1, padding='same', bias=False)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding='same', bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool1d(5, stride=2)

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu0(out)
        out = self.conv1(out)
        out += x
        out = self.relu1(out)
        out = self.pooling(out)
        return out
import torch.nn as nn
import torch

# define architecture
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(5, 5)):
        super(Block, self).__init__()
        self.conv0 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=(2, 2), bias=False)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel, stride=(2, 2), bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool1d(5, stride=(2, 2))
        self.flat = nn.Flatten()

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu0(out)
        out = self.conv1(out)
        out += x
        out = self.relu1(out)
        out = self.pooling(out)
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5),
            Block(32, 64),
            Block(64, 256),
            Block(256, 512),
            Block(512, 1024),
            Block(1024, 2048)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(2048, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
            nn.Softmax(1),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layer(x)
        return x


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
import torch
from torch.utils.data import Dataset, DataLoader


class ECG(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        X = self.X.iloc[idx].values[None, ...]
        y = self.y.iloc[idx]
        return X, y

    def __len__(self):
        return len(self.y)


import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

train = pd.read_csv('mitbih_train.csv', header=None)
test = pd.read_csv('mitbih_test.csv', header=None)


X_train, y_train = train.iloc[:, :187], train[187]
X_test, y_test = test.iloc[:, :187], test[187]
class ECG(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        data = torch.tensor(sample.values[:-1], dtype=torch.float32)
        label = torch.tensor(sample.values[-1], dtype=torch.long)

        return data, label

    def __len__(self):
        return len(self.df)


train_loader = DataLoader(ECG(df=X_train), batch_size=32, shuffle=True)
test_loader = DataLoader(ECG(df=X_test), batch_size=32, shuffle=False)
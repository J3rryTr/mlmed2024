import pandas as pd
import torch
import torch.nn as nn
from utils.data_loading import ECG
from torch.utils.data import DataLoader
from train import train_test
from CNN.cnn_custom import CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = pd.read_csv('/home/jerry/code/MLmedicine/lab1/dataset/mitbih_train.csv', header=None)
test = pd.read_csv('/home/jerry/code/MLmedicine/lab1/dataset/mitbih_test.csv', header=None)

X_train, y_train = train.iloc[:, :187], train[187]
X_test, y_test = test.iloc[:, :187], test[187]

train_loader = DataLoader(ECG(X=X_train, y=y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(ECG(X=X_test, y=y_test), batch_size=32, shuffle=False)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


train_test(
    model=model,
    train_loader=train_loader, test_loader=test_loader,
    epochs=100, criterion=criterion,
    optimizer=optimizer, device=device
).run()
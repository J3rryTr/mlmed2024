import train
import dataset
import model

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train.Training(model=model, train_loader=dataset.train_loader, test_loader=dataset.test_loader, epochs=3, criterion=model.criterion, optimizer=model.optimizer, device=device, save_path='/home/jerry/code/FaceDetection/output').train()
from tqdm import tqdm
import torch


class train_test():
    def __init__(self, model, train_loader, test_loader, epochs , criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs

    def run(self):
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in tqdm(self.train_loader):
                # print(inputs)
                inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs).float()

                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss.append(running_loss/len(self.train_loader))
            train_acc.append(100.*correct/total)

            self.model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(self.test_loader):
                    #print(inputs)
                    inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs).float()

                    loss = self.criterion(outputs, labels)

                    self.optimizer.step()
                    running_loss += loss.item()

                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            test_loss.append(running_loss / len(self.test_loader))
            test_acc.append(100. * correct / total)

            print(f'Epoch: {epoch+1}/{self.epochs}, '
                  f'Training Accuracy: {train_acc[-1]:.2f}%, Training Loss: {train_loss[-1]:.4f}, '
                  f'Test Accuracy: {test_acc[-1]:.2f}%, Test Loss: {test_loss[-1]:.4f}')
        return train_loss, train_acc, test_loss, test_acc
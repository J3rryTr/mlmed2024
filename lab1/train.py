import model
import torch
import torch.optim as optim


class Training():
    def __init__(self, model, train_loader, test_loader, epochs, criterion, optimizer, device, save_path):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.save_path = save_path

    def train(self):
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
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
                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            test_loss.append(running_loss/len(self.test_loader))
            test_acc.append(100.*correct/total)

            print(f'Epoch: {epoch+1}/{self.epochs}, Training Loss: {train_loss[-1]:.4f}, Training Accuracy: {train_acc[-1]:.2f}%, Test Loss: {test_loss[-1]:.4f}, Test Accuracy: {test_acc[-1]:.2f}%')

            torch.save(self.model.state_dict(), f"{self.save_path}_epoch{epoch + 1}.pth") #save weight

        return train_loss, train_acc, test_loss, test_acc
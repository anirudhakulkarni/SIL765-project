import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# define the CNN model
class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(640, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x = self.flatten(x)
        # print("x: ", x.shape)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x

# define the custom dataset
class PacketDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CNNClassifier:
    def __init__(self, num_classes,X_train, y_train, X_test, y_test):
        self.num_classes = num_classes
        X_train = self.pad(X_train)
        X_test = self.pad(X_test)
        y_train = torch.tensor(y_train.values.astype(np.int64))
        y_test = torch.tensor(y_test.values.astype(np.int64))
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dataset = PacketDataset(X_train, y_train)
        self.input_size = len(X_train[0])
        self.model = CNN(self.input_size, self.num_classes)
        print("Model: ", self.model)
        print("input_size: ", self.input_size)
        
    def pad(self, X):
        X = [torch.from_numpy(x).float() for x in X]
        # keep max sequence length = 100
        X = [x[:100] for x in X]
        # print(f'mean: {np.mean([len(x) for x in X])}, median: {np.median([len(x) for x in X])}, max: {np.max([len(x) for x in X])}')
        X = pad_sequence(X, batch_first=True)
        return X
    
    def train(self):
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.num_epochs = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        for epoch in range(self.num_epochs):
            for data, labels in self.dataloader:
                # data = data.permute(1,0)
                data = data.unsqueeze(1)
                # print(data.shape)
                data = data.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.4f}')
            # evaluate the model on the test set
            with torch.no_grad():
                total_correct = 0
                total_ = 0
                for data, labels in self.dataloader:
                    data = data.unsqueeze(1)
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_ += labels.size(0)
                print(f'Epoch {epoch+1}/{self.num_epochs}, Accuracy: {total_correct/total_}')
        return self.model

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            total_correct = 0
            total_ = 0
            for data, labels in self.dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_ += labels.size(0)
            print(f'Accuracy: {total_correct/total_}')

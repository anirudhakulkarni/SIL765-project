import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# define the CNN model
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self, input_shape, classes):
        super(CNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.elu1 = nn.ELU(alpha=1.0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.elu2 = nn.ELU(alpha=1.0)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=4, padding=2)
        self.dropout1 = nn.Dropout(p=0.1)
        
        # Block 2
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=1, padding=4)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.relu1 = nn.ReLU()
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding=4)
        self.bn4 = nn.BatchNorm1d(num_features=64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=4, padding=2)
        self.dropout2 = nn.Dropout(p=0.1)
        
        # Block 3
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding=4)
        self.bn5 = nn.BatchNorm1d(num_features=128)
        self.relu3 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=4)
        self.bn6 = nn.BatchNorm1d(num_features=128)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=8, stride=4, padding=2)
        self.dropout3 = nn.Dropout(p=0.1)
        
        # Block 4
        self.conv7 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8, stride=1, padding=4)
        self.bn7 = nn.BatchNorm1d(num_features=256)
        self.relu5 = nn.ReLU()
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=8, stride=1, padding=4)
        self.bn8 = nn.BatchNorm1d(num_features=256)
        self.relu6 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=8, stride=4, padding=2)
        self.dropout4 = nn.Dropout(p=0.1)
        
        # Fully connected layers
        
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.bn9 = nn.BatchNorm1d(num_features=512)
        self.relu7 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.bn10 = nn.BatchNorm1d(num_features=512)
        self.relu8 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=512, out_features=1500)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu1(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu3(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu4(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu5(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu6(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        # print(x.shape)
        x = self.fc1(x)
        x = self.bn9(x)
        x = self.relu7(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        x = self.bn10(x)
        x = self.relu8(x)
        x = self.dropout6(x)
        x = self.fc3(x)
        x = self.softmax(x)
        
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
        print("Train data shape: ", X_train.shape)
        print("Train labels shape: ", y_train.shape)
        print("Test data shape: ", X_test.shape)
        print("Test labels shape: ", y_test.shape)
        
        
    def pad(self, X):
        X = [torch.from_numpy(x).float() for x in X]
        # keep max sequence length = 100
        # write length as 101th element
        
        # X = [x[:100] for x in X]
        # print(f'mean: {np.mean([len(x) for x in X])}, median: {np.median([len(x) for x in X])}, max: {np.max([len(x) for x in X])}')
        X = pad_sequence(X, batch_first=True)
        return X
    
    def train(self):
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.num_epochs = 100
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        for epoch in tqdm(range(self.num_epochs)):
            for data, labels in self.dataloader:
                # data = data.permute(1,0)
                data = data.unsqueeze(1)
                # print(data.shape)
                data = data.to(self.device)
                labels = labels.to(self.device)
                # print(labels.min(), labels.max())
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

    def predict(self):
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

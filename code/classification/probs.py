import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def run(path, gpu):
    # Read the pickle object
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data)

    # Encode the class labels
    label_encoder = LabelEncoder()
    df['class_label'] = label_encoder.fit_transform(df['class_label'])

    # Define the maximum sequence length
    max_len = 0
    for i in df['lengths']:
        max_len=max(max_len,len(i))

    # Pad the sequences to the maximum length
    sequences = np.array(df['lengths'])
    padded_sequences = np.zeros((len(sequences), max_len))
    for i, sequence in enumerate(sequences):
        padded_sequences[i, :len(sequence)] = sequence

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(padded_sequences, df['class_label'], test_size=0.2)

    # Convert the data to PyTorch tensors
    X_train = torch.from_numpy(X_train).to(torch.float32)
    y_train = torch.from_numpy(y_train.values).long()
    X_val = torch.from_numpy(X_val).to(torch.float32)
    y_val = torch.from_numpy(y_val.values).long()
    # check if split is stratified
    # print("Training set class distribution:")
    # print(y_train.unique(return_counts=True)) 
    # print("Validation set class distribution:")
    # print(y_val.unique(return_counts=True))
    print(y_train.unique().shape)
    print(y_val.unique().shape)

    # Define a custom dataset
    class SequenceDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, index):
            sequence = self.sequences[index]
            label = self.labels[index]
            return sequence, label



    # Initialize the model
    input_dim=0
    for i in X_train:
        input_dim = max(input_dim,len(i))
    # input_dim = len(X_train[0])
    hidden_dim = 128
    output_dim = 1500
    learning_rate = 1e-3
    print(input_dim)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Define the dataloaders
    train_dataset = SequenceDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = SequenceDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Define the model architecture
    class RNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, batch_size):
            super(RNN, self).__init__()
            self.batch_size=batch_size
            # simple 1d conv 
            self.conv1 = nn.Conv1d(1, 8, 3, padding=1)
            self.bn1 = nn.BatchNorm1d(8)
            self.relu = nn.ReLU()        
            
            # parameters for the RNN
            self.hidden_dim = hidden_dim
            self.input_dim = input_dim
            self.output_dim = output_dim

            # define the RNN
            # self.rnn = nn.LSTM(8*input_dim, hidden_dim, batch_first=True)
            self.rnn = nn.RNN(8*input_dim, hidden_dim, batch_first=True)
            # define the output layer
            self.fc = nn.Linear(hidden_dim, output_dim)    
            self.h = self.init_hidden(self.batch_size)    
            self.flatten = nn.Flatten()
            # self.fc = nn.Linear(8*input_dim, output_dim)
            
        def init_hidden(self, bsz):
            # initialize the hidden state
            self.h=torch.zeros(1, bsz, self.hidden_dim).to(device)
        
        def forward(self, x):
            # reshape the input
            # x = x.view(self.batch_size, self.input_dim, -1)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.flatten(x)
            # initialize the hidden state
            self.init_hidden(x.shape[0])
            # get RNN unit outputs
            out, h = self.rnn(x.unsqueeze(1), self.h)
            # get the output for the last time step
            out = self.fc(out[:, -1, :])
            return out

    device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    rnn = RNN(input_dim, hidden_dim, output_dim, batch_size=64)
    rnn = rnn.to(device)
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)


    from sklearn.metrics import precision_recall_fscore_support
    def get_precision_recall_f1(y_true, y_pred):
        """Calculate precision, recall, F1 score for each class.
        Args:
            y_true (list): list of true labels
            y_pred (list): list of predicted labels
        Returns:
            dict: dictionary with precision, recall, F1 score for each class
        """
        # Calculate precision, recall, F1 score for each class
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    num_epochs = 30
    # optimizer = optimizer.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        rnn.train()
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            labels = labels.to(device)
            sequences = sequences.to(device)
            sequences = sequences.unsqueeze(1)
            outputs = rnn(sequences)
            
            # print(sequences.shape)
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * sequences.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() /len(train_dataset)
        print('Train Loss: {:.4f} Train Acc: {:.4f} %'.format(epoch_loss, epoch_acc*100))
        
        # test the model
        rnn.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            total_preds = []
            total_labels = []
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                sequences = sequences.unsqueeze(1)
                outputs = rnn(sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_preds.append(predicted)
                total_labels.append(labels)
            
            print('Epoch: {}, Test Accuracy of the model on the test sequences: {} %'.format(epoch, 100 * correct / total))
            total_preds = torch.cat(total_preds, dim=0)
            total_labels = torch.cat(total_labels, dim=0)
            print(get_precision_recall_f1(total_labels.cpu(), total_preds.cpu()))

    # save model
    # torch.save(rnn.state_dict(), 'rnn_model_.ckpt')

    # evaluate on test set and report the results
    rnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        total_preds = []
        total_labels = []
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            sequences = sequences.unsqueeze(1)
            outputs = rnn(sequences)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_preds.append(predicted)
            total_labels.append(labels)
        total_preds = torch.cat(total_preds, dim=0)
        total_labels = torch.cat(total_labels, dim=0)
        
        return {'file':path, 'test_acc': 100 * correct / total, 'precision': get_precision_recall_f1(total_labels.cpu(), total_preds.cpu())['precision'], 'recall': get_precision_recall_f1(total_labels.cpu(), total_preds.cpu())['recall'], 'f1': get_precision_recall_f1(total_labels.cpu(), total_preds.cpu())['f1']}
import os

if __name__ == '__main__':
    # run on the all files present in directory="../../dataset/pickles/probs/"
    # and save the results in csv
    
    files = os.listdir("../../dataset/pickles/select")
    files = [file for file in files if file.endswith(".pickle")]
    print(files)
    results = []
    gpuid = 0
    maxgpu=7
    # create a new process for each file so that we can run multiple files in parallel
    import multiprocessing
    pool = multiprocessing.Pool(processes=8)
    for file in files:
        print(file,gpuid)
        results.append(pool.apply_async(run, args=(os.path.join("../../dataset/pickles/select/", file), gpuid)))
        gpuid = (gpuid+1)%maxgpu
    pool.close()
    pool.join()
    final_results=[]
    for p in results:
        try:
            results.append(p.get())
        except:
            pass
    # results = [p.get() for p in results]
    print(results)
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("results_probs.csv")
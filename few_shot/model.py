'''
Author: Qi7
Date: 2023-04-06 21:32:59
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-04 13:47:42
Description: deep learning models definition
'''
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, seq_num, num_class):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_size, 
            hidden_size = 256,
            num_layers = 1,
            batch_first = True
        )
        self.num_class = num_class
        self.dropout = nn.Dropout(0.2)
        #self.linear = nn.Linear(512, self.num_class)
        self.linear = nn.Linear(256, self.num_class)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        outputs = self.linear(self.dropout(x))
        # multiclass classification
        if self.num_class > 1:
            outputs = self.sigmoid(outputs)
        return outputs

class ResBlock(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, 
                 kernel_size, stride, n_pad, dropout=0.2):
        super(ResBlock, self).__init__()
        self.batchnorm1 = nn.BatchNorm1d(n_input_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(n_input_channels, n_output_channels, kernel_size,
                               stride = stride, padding = n_pad)
        self.batchnorm2 = nn.BatchNorm1d(n_output_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_output_channels, n_output_channels, kernel_size,
                               stride = 1, padding = 'same')
        self.max_pooling = nn.MaxPool1d(stride)

    def forward(self, x):
        out = self.batchnorm1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv1(out)
        out = self.batchnorm2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.conv2(out)
        if out.shape != x.shape:
            # if x.shape[-1] != out.shape[-1]:
            #     x = F.pad(x, pad = (1,1))
            x = self.max_pooling(x)
        out = out + x
        return out

class QNN(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, 
                 kernel_size, stride, n_classes, dropout=0.5):
        super(QNN, self).__init__()
        self.conv1 = nn.Conv1d(n_input_channels, n_output_channels, kernel_size = 7,
                               stride = stride, padding = 'same')
        self.batchnorm1 = nn.BatchNorm1d(n_output_channels)
        self.relu1 = nn.ReLU()
        self.max_pooling = nn.MaxPool1d(2)
        self.resblock1 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 1, 
                                 dropout=dropout)
        self.resblock2 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 1, 
                                 dropout=dropout)
        self.resblock3 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 1, 
                                 dropout=dropout)
        self.resblock4 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 0, 
                                 dropout=dropout)
        self.resblock5 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 1, 
                                 dropout=dropout)
        self.resblock6 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 0, 
                                 dropout=dropout)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(64 * 15), 128)
        self.fc1 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.max_pooling(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.fc1(x)
        return x


class SiameseNet(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, 
                 kernel_size, stride, n_classes, dropout=0.5) -> None:
        super(SiameseNet, self).__init__()
        self.conv1 = nn.Conv1d(n_input_channels, n_output_channels, kernel_size = 7,
                               stride = stride, padding = 'same')
        self.batchnorm1 = nn.BatchNorm1d(n_output_channels)
        self.relu1 = nn.ReLU()
        self.max_pooling = nn.MaxPool1d(2)
        self.resblock1 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 1, 
                                 dropout=dropout)
        self.resblock2 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 1, 
                                 dropout=dropout)
        self.resblock3 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 1, 
                                 dropout=dropout)
        self.resblock4 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 0, 
                                 dropout=dropout)
        self.resblock5 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 1, 
                                 dropout=dropout)
        self.resblock6 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 0, 
                                 dropout=dropout)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(64 * 15), n_classes)
        # self.fc2 = nn.Linear(int(64 * 15), 2)
        self.fc3 = nn.Linear(int(64 * 15), 100)
    
    def forward_once(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.max_pooling(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.flatten(x)
        x_classification = self.fc(x)
        x_embedding = self.fc3(x)
        return x_embedding, x_classification

    def forward(self, input1, input2):
        # forward pass through both branches of the network
        output1_embedding, output1_classification = self.forward_once(input1)
        output2_embedding, output2_classification = self.forward_once(input2)
        return output1_embedding, output1_classification, output2_embedding, output2_classification


class DistanceNet(nn.Module):
    def __init__(self, embedding_net) -> None:
        super(DistanceNet, self).__init__()
        self.embedding_net = embedding_net
        self.trainlayer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,2)
        )
    
    def forward(self, x1, x2):
        # freeze the parameters update for embedding net
        with torch.no_grad():
            x1 = self.embedding_net(x1)
            x2 = self.embedding_net(x2)
            
        output1 = self.trainlayer(x1)
        output2 = self.trainlayer(x2)
        return output1, output2
    
    def get_embedding(self, x):
        with torch.no_grad():
            x = self.embedding_net(x)
            
        output = self.trainlayer(x)
        return output
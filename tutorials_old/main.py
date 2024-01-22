import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from pathlib import Path

import influxdb_client
import tsai
#from tsai.all import *

import re
import pytz
from datetime import datetime

import enum
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
#from data_loader import SiameseNetworkDataset
#from models import LSTM
#from customized_criterion import ContrastiveLoss
from torchinfo import summary
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold, train_test_split

import tutorials_old.load_data as load_data
import few_shot.loader as loader
import few_shot.model as models
import few_shot.training as training

p = Path('.')
save_eval_path = p / "evaluation_results"
save_model_path = p / "saved_models"

x = load_data.load('X.npy')
print("x type is ",type(x))
print("shape of x is ",x.shape)
y = load_data.load('y.npy')

# normalization on input data x
x = (x - x.mean(axis=0)) / x.std(axis=0)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

#training_set = np.concatenate((X_train, y_train.reshape(-1,1)) ,axis=1)
#test_set = np.concatenate((X_test, y_test.reshape(-1,1)) ,axis=1)

X_train = np.transpose(X_train,(0,2,1))
X_test = np.transpose(X_test,(0,2,1))

#y_train = np.expand_dims(y_train,axis=1)
#y_test = np.expand_dims(y_test,axis=1)

training_set = loader.waveformDataset(X_train, y_train)
test_set = loader.waveformDataset(X_test, y_test)

#X = np.transpose(X, (0,2,1)) # transpose to match the lstm standard
#y = np.expand_dims(y, axis=1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=7)
trainset = training_set
testset = test_set

# Hyper parameters
batch_size = 256
learning_rate = 0.001
num_epochs = 500
history = dict(train_loss=[], test_loss=[], train_acc=[], test_acc=[], train_f1=[], test_f1=[], test_f1_all=[])

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#device = "cpu"

trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
testloader = DataLoader(testset, shuffle=False, batch_size=1024) # get all the samples at once
model = models.LSTM(input_size=6, seq_num=2000, num_class=8)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# model training

training.model_train_multiclass(
    model=model,
    train_loader=trainloader,
    val_loader=testloader,
    num_epochs=num_epochs,
    optimizer=optimizer,
    device=device,
    history=history
)

torch.save(model.state_dict(), save_model_path / f"multiclass_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_best_model.pth")
np.save(save_eval_path / f"multiclass_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_history.npy", history)
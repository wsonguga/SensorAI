'''
Author: Qi7
Date: 2023-05-23 17:06:32
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-02 15:08:11
Description: training the SNN model
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.nn as nn
import copy
import tqdm, time

from loader import tripletDataset
from model import SiameseNet
from training import model_train_multiclass

#save_model_path = "saved_models/new_snn/"
#X = np.load('dataset/8cases_jinan/new_training_set/X_norm.npy')
# = np.load('dataset/8cases_jinan/new_training_set/y.npy')

from pathlib import Path
#import load_data as ld

p = Path('.')
datapath = p / "AI_engine/test_data/"

save_eval_path = p / "AI_engine/evaluation_results/"
save_model_path = p / "AI_engine/saved_models/"
datapath = p / "AI_engine/test_data/"

#data = np.load(datapath/'X.npy')#ld.selectFileAndLoad()
#print("shape of  data is ",data.shape)

#X = data[:, :data.shape[1]-1]  # data
#y = data[:, -1] # label

X = np.load(datapath/'X.npy')
y = np.load(datapath/'y.npy')
#save_model_path = p

X_train, X_cv, y_train, y_cv = train_test_split(X, y, train_size=0.75, test_size=0.25, shuffle=True, random_state=27)
trainset = tripletDataset(X_train, y_train)
validset = tripletDataset(X_cv, y_cv)

# Hyper parameters
batch_size = 128
learning_rate = 0.001
num_epochs = 20
history = dict(test_loss=[], train_loss=[], test_acc=[], test_f1=[], test_f1_all=[])

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(validset, batch_size=batch_size, shuffle=True)

model = SiameseNet(
            n_input_channels=6,
            n_output_channels=64,
            kernel_size=3,
            stride=1,
            n_classes=8
        )

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
margin = 2 # parameters for penalizing
triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
classification_loss = nn.CrossEntropyLoss()
best_loss = np.inf
best_weights = None

start = time.time()
for epoch in range(num_epochs):
    # training
    model.train()
    total_loss = 0.0
    # for batch_idx, (sample1, sample2, sample3, label_pos, label_neg) in tqdm.tqdm(data_loader, desc=f"Epoch {epoch}"):
    for batch_idx, (sample1, target1, sample2, sample3, label_pos, label_neg) in enumerate(data_loader):
        sample1, sample2, sample3 = sample1.to(device), sample2.to(device), sample3.to(device)
        label_anchor, label_pos, label_neg = target1.to(device, dtype=torch.long), label_pos.to(device), label_neg.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output1_e, output1_c, output2_e, output2_c = model(sample1, sample2)
        _, _, output3_e, output3_c = model(sample1, sample3)
        
        # classification loss
        loss_classification = classification_loss(output1_c, label_anchor)
        # triple loss
        loss_contrastive = triplet_loss(output1_e, output2_e, output3_e)

        combined_loss = 0 * loss_classification + loss_contrastive
        # Backward pass and optimization
        combined_loss.backward()
        optimizer.step()
        
        total_loss += combined_loss.item()
        
        if batch_idx % 20 == 0:
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, batch_idx+1, len(data_loader), loss_contrastive.item()))
            
    history['train_loss'].append(total_loss / len(data_loader))

    print('Epoch [{}/{}], Average Train Loss: {:.4f}'
          .format(epoch+1, num_epochs, total_loss / len(data_loader)))

    
    # validation
    model.eval()
    total_loss = 0.0
    for batch_idx, (sample1, target1, sample2, sample3, label_pos, label_neg) in enumerate(test_data_loader):
        sample1, sample2, sample3 = sample1.to(device), sample2.to(device), sample3.to(device)
        label_anchor, label_pos, label_neg = target1.to(device, dtype=torch.long), label_pos.to(device), label_neg.to(device)
        
        output1_e, output1_c, output2_e, output2_c = model(sample1, sample2)
        _, _, output3_e, output3_c = model(sample1, sample3)
        
        loss_classification = classification_loss(output1_c, label_anchor)
        loss_contrastive = triplet_loss(output1_e, output2_e, output3_e)
        combined_loss = loss_classification + 0.2 * loss_contrastive
        total_loss += combined_loss.item()
    
    test_loss = total_loss / len(test_data_loader)
    if best_loss > test_loss:
        best_loss = test_loss
        best_weights = copy.deepcopy(model.state_dict())
    history['test_loss'].append(test_loss)

    print('Epoch [{}/{}], Average Test Loss: {:.4f}'
          .format(epoch+1, num_epochs, test_loss))

end = time.time()
print(f"Training is done. Total time: {end - start} seconds")

torch.save(model.state_dict(), save_model_path + f"new_2_loss_last_round_2d_snn_margin{margin}_8cases_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_best_model.pth")
model.load_state_dict(best_weights)
torch.save(model.state_dict(), save_model_path + f"new_2_loss_2d_snn_margin{margin}_8cases_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_best_model.pth")
np.save(save_model_path + f"new_2_loss_2d_snn_margin{margin}_8cases_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_history.npy", history)




# # Set up the Siamese network, optimizer, and loss function
# siamese_net = SiameseNet()
# optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)
# margin = 1.0

# # Set device to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# siamese_net.to(device)

# # Prepare your data and targets
# train_data = ...
# train_targets = ...

# # Create the Siamese dataset and data loader
# siamese_dataset = SiameseDataset(train_data, train_targets)
# data_loader = DataLoader(siamese_dataset, batch_size=64, shuffle=True)

# # Training loop

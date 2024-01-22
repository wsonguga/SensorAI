'''
Author: Qi7
Date: 2023-04-07 10:41:35
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-02 21:38:38
Description: dataloader definition
'''
from torch.utils.data import Dataset
import torch
import numpy as np

class waveformDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        features = self.X[idx]
        target = self.y[idx]
        return features, target
    

class tripletDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __getitem__(self, index):
        sample1, target1 = self.data[index], self.targets[index]
        positive_indices = torch.where(self.targets == target1)[0]
        negative_indices = torch.where(self.targets != target1)[0]
        sample2 = self.data[positive_indices[torch.randint(len(positive_indices), (1,))]][0]
        sample3 = self.data[negative_indices[torch.randint(len(negative_indices), (1,))]][0]
        return sample1, target1, sample2, sample3, torch.Tensor([1]), torch.Tensor([0])
    
    def __len__(self):
        return len(self.data)

class siameseDataset(Dataset):
    def __init__(self, data, targets) -> None:
        self.data = data
        self.targets = targets
        self.labels_set = set(self.targets)
        self.label_to_indices = {label: np.where(self.targets == label)[0]
                                 for label in self.labels_set}
    
    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        sample1, label1 = self.data[index], self.targets[index]
        
        # same category
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = np.random.choice(list(self.labels_set - set([label1])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        
        sample2 = self.data[siamese_index]
        
        return (sample1, sample2), target
    
    def __len__(self):
        return len(self.data)
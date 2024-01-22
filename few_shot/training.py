'''
Author: Qi7
Date: 2023-04-07 11:13:41
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-04 14:46:35
Description: helper function for training the model
'''
import numpy as np
import copy
import torch
import torch.nn as nn 
import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def model_train(model, train_loader, val_loader, num_epochs, optimizer, device, history):
    """_summary_

    Args:
        model (_type_): _description_
        train_loader (_type_): _description_
        val_loader (_type_): _description_
        num_epochs (_type_): _description_
        optimizer (_type_): _description_
        device (_type_): _description_
        history (_type_): the statistic information such as accuracy, f1-score, loss

    Returns:
        _type_: statistical information. history
    """    
    loss_fn = nn.BCELoss()
    
    best_f1 = -np.inf
    best_weights = None
    
    for epoch in range(num_epochs):
        train_acc, train_loss = [], []
        model.train()
        for X_batch, y_batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}"):
            # forward pass
            X_batch = X_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.float32)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            acc = (y_pred.round() == y_batch).float().mean()
            
            # store metrics
            train_loss.append(float(loss))
            train_acc.append(float(acc))
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # update weights
            optimizer.step()
        
        # evaluate at end of each epoch
        model.eval()
        for X_val, y_val in val_loader:
            X_val = X_val.to(device, dtype=torch.float32)
            y_val = y_val.to(device, dtype=torch.float32)
            y_pred = model(X_val)
            acc = (y_pred.round() == y_val).float().mean()
            f1 = f1_score(y_true=y_val.cpu().detach().numpy(), 
                          y_pred=y_pred.round().cpu().detach().numpy(), 
                          average='binary')
            
            history['train_loss'].append(sum(train_loss) / len(train_loss))
            history['train_acc'].append(sum(train_acc) / len(train_acc))
            history['test_acc'].append(float(acc))
            history['test_f1'].append(float(f1))
            print(f"train acc: {history['train_acc'][-1]}. val acc: {history['test_acc'][-1]}")
            print(f"train loss: {history['train_loss'][-1]}")
            print(f"val f1: {history['test_f1'][-1]}")
            
            
            if f1 > best_f1:
                best_f1 = f1
                best_weights = copy.deepcopy(model.state_dict())
                
            # only validate one batch
            break
    
    print(f"Training Done. best F1 score: {best_f1}")
    # save the model with best performance
    model.load_state_dict(best_weights)
    
    return 0


def model_train_multiclass(model, train_loader, val_loader, num_epochs, optimizer, device, history):
    """_summary_
        model training function for multi-class classification
    Args:
        model (_type_): _description_
        train_loader (_type_): _description_
        val_loader (_type_): _description_
        num_epochs (_type_): _description_
        optimizer (_type_): _description_
        device (_type_): _description_
        history (_type_): the statistic information such as accuracy, f1-score, loss

    Returns:
        _type_: statistical information. history
    """
    
    loss_fn = nn.CrossEntropyLoss()
    
    best_f1 = -np.inf
    best_weights = None
    
    for epoch in range(num_epochs):
        train_loss, predicted_labels, ground_truth_labels = [], [], []
        model.train()
        for X_batch, y_batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}"):
            # forward pass
            X_batch = X_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.long)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            _, predicted = torch.max(y_pred, 1) # get the index of the predicated label, which represent the predicted class number
            predicted_labels.append(predicted.cpu().detach().numpy())
            ground_truth_labels.append(y_batch.cpu().detach().numpy())
            # store metrics
            train_loss.append(float(loss))            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
        
        # evaluate at end of each epoch
        model.eval()
        for X_val, y_val in val_loader:
            X_val = X_val.to(device, dtype=torch.float32)
            y_val = y_val.to(device, dtype=torch.long)
            y_pred = model(X_val)
            loss = loss_fn(y_pred, y_val)
            _, predicted = torch.max(y_pred, 1)
            predicted = predicted.cpu().detach().numpy()
            y_val = y_val.cpu().detach().numpy()
            
            val_acc = accuracy_score(predicted, y_val)
            val_f1 = f1_score(predicted, y_val, average='macro')
            val_f1_none = f1_score(predicted, y_val, average=None)
            
            predicted_labels = np.concatenate(predicted_labels).ravel()
            ground_truth_labels = np.concatenate(ground_truth_labels).ravel()
            
            history['train_loss'].append(sum(train_loss) / len(train_loss))
            history['test_loss'].append(float(loss))
            history['train_acc'].append(accuracy_score(predicted_labels, ground_truth_labels))
            history['train_f1'].append(f1_score(predicted_labels, ground_truth_labels, average='weighted'))
            history['test_acc'].append(float(val_acc))
            history['test_f1'].append(float(val_f1))
            history['test_f1_all'].append(val_f1_none.tolist())
            
            print("-------------------------------------------")
            print(f"train loss: {history['train_loss'][-1]}. test loss: {history['test_loss'][-1]}\n")
            print(f"train acc: {history['train_acc'][-1]}. val acc: {history['test_acc'][-1]}")
            print(f"train f1: {history['train_f1'][-1]}. val f1: {history['test_f1'][-1]}")
            print(f"test f1 all: {history['test_f1_all'][-1]}")
            
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_weights = copy.deepcopy(model.state_dict())
                
            # only validate one batch
            break
    
    print(f"Training Done. best F1 score: {best_f1}")
    # save the model with best performance
    model.load_state_dict(best_weights)
    
    return 0

def evaluate(dataloader, model, loss_fn, device, history):
    model.eval()
    predicted_labels, ground_truth_labels = np.array([], dtype=int), np.array([], dtype=int)
    loss_curve = []
    for X_val, y_val in dataloader:
        X_val = X_val.to(device, dtype=torch.float32)
        y_val = y_val.to(device, dtype=torch.long)
        y_pred = model(X_val)
        loss = loss_fn(y_pred, y_val)
        _, predicted = torch.max(y_pred, 1)
        predicted = predicted.cpu().detach().numpy()
        y_val = y_val.cpu().detach().numpy()
        
        predicted_labels = np.concatenate((predicted_labels, predicted)).ravel()
        ground_truth_labels = np.concatenate((ground_truth_labels, y_val)).ravel()
        # loss_curve.append(float(loss.cpu().detach().numpy()))
        loss_curve.append(float(loss))
        
    val_acc = accuracy_score(predicted_labels, ground_truth_labels)
    val_f1 = f1_score(predicted_labels, ground_truth_labels, average='macro')
    val_f1_none = f1_score(predicted_labels, ground_truth_labels, average=None)
    history['test_acc'].append(float(val_acc))
    history['test_f1'].append(float(val_f1))
    history['test_f1_all'].append(val_f1_none.tolist())
    history['test_loss'].append(sum(loss_curve) / len(loss_curve))
    
    
    
def fit(train_loader, val_loader, model, loss_fn, optimizer, n_epochs, device, history, save_model_path, margin, scheduler=None):
    max_val_loss = 1
    for epoch in range(n_epochs):
        model.train()
        losses = []
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
                
            data = tuple(d.to(device, dtype=torch.float32) for d in data)
            target = target.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs = model(*data)
            
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target
                
            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        total_loss /= (batch_idx + 1)
        history['train_loss'].append(total_loss)
        
        print(f"Epoch: {epoch + 1}/{n_epochs}")
        print(f"train loss: {total_loss}")
            
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
                
            data = tuple(d.to(device, dtype=torch.float32) for d in data)
            target = target.to(device, dtype=torch.float32)
            
            outputs = model(*data)
            
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target
                
            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            val_loss += loss.item()
            
        total_loss = val_loss / (batch_idx + 1)
        history['val_loss'].append(total_loss)
        if max_val_loss > total_loss:
            max_val_loss = total_loss
            torch.save(model.state_dict(), save_model_path + f"margin_{margin}_epoch_{n_epochs}_contrastive_model.pth")
        
        print(f"val loss: {total_loss}")
        
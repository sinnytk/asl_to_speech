#basic imports
import os
import glob
import time 
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#Deep learning Imports
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms as T
import torch.optim as optim

#custom scripts import
from model import Net
import HandSign
from mappings import label_mappings
from pre_trained_model import initialize_model,set_parameter_requires_grad
from torch.utils.data import Dataset, DataLoader

    
def acc_score( pred, actual):
    matches = [i == j.item() for i,j in zip(pred, actual)]
    return matches.count(True)/len(matches)


def train(dataset,model_name,split,batch_size=128,epochs=15,test_interval=5):

    # if model_name == 'custom_model':
    #     print('working')
    model = Net(kernel_size=2)
    model.load_state_dict(torch.load('custom_model_0.pt', map_location='cuda:0'))
    # else:
    #     print('this working too')
    #     model,input_size = initialize_model(model_name=model_name, num_classes=29, feature_extract=False, use_pretrained=True)

    print(model)
    history={'train':[],'val':[]}
    train_size = int(len(dataset)*split)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset,[train_size,test_size]) 

    train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data,batch_size=batch_size//8, shuffle=True)


    print(f"train size :{train_size}")
    print(f"test size :{test_size}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'On device: {torch.cuda.get_device_name(0)}')    

    model.to(device)

    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    loss_function.to(device)
    t_progress_bar =  tqdm(train_loader)

    for epoch in range(epochs):
        model.train()
        acc = 0.0
        loss = 0.0
        for inputs,labels in t_progress_bar:
            inputs = Variable(inputs.float().to(device), requires_grad=True)
            labels = Variable(labels.to(device)).long()
            optimizer.zero_grad()
            outputs = model(inputs)                       
            batch_loss = loss_function(outputs, torch.argmax(labels,dim=1))
            batch_loss.backward()
            optimizer.step()

            local_acc = acc_score(torch.argmax(outputs,dim=1), torch.argmax(labels,dim=1))
            acc += local_acc
            loss += batch_loss.item()

            t_progress_bar.set_description(f'EPOCH: {epoch},TRAIN_ACC: {round(local_acc,4)},TRAIN_LOSS: {round(batch_loss.item(),4)}')
            t_progress_bar.refresh()

        for i,l in test_loader:        
            i = Variable(i.float().to(device), requires_grad=True)
            l = Variable(l.to(device)).long()
            model.eval()
            val_output = model(i)
            val_loss = loss_function(val_output,torch.argmax(l,dim=1))
            val_acc = acc_score(torch.argmax(val_output,dim=1),torch.argmax(l,dim=1))
            print(f'EPOCH: {epoch},VAL_ACC: {round(val_acc,4)},VAL_LOSS: {round(val_loss.item(),4)}')
            break
        
        torch.save(model.state_dict(),f'{model_name}_{epoch}.pt')
        
if __name__ == "__main__":

    # model,input_size = initialize_model(model_name='densenet', num_classes=29, feature_extract=False, use_pretrained=True)
    dataset = HandSign.HandSign(img_size=100)
    train(dataset,model_name='custom_model',split=0.7,epochs=3,batch_size=128,test_interval=5)


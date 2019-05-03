import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import numpy as np
import argparse
import pdb
import time
from tensorboardX import SummaryWriter
import os

from Datasets import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--debug', nargs='?', const=1, type=bool, default=False)
parser.add_argument('--data_path', nargs='?', const=1, type=str, default='/home/kimdy/code/spdl19/ssl_data_96')
parser.add_argument('--save_path', nargs='?', const=1, type=str, default='./')
parser_args = parser.parse_args()

data_path = parser_args.data_path
save_path = parser_args.save_path
log_dir = save_path+'logs/'
os.makedirs(save_path+'models/',exist_ok=True)
model_path = save_path+'models/'

num_epochs = 100
learning_rate = 1e-3
batch_size = 128

trainset = dataLabel(64, data_path, train=True)
train_loader = DataLoader(trainset, batch_size=batch_size)
print('Train Set Loaded!')
validset = dataLabel(64, data_path, train=False)
valid_loader = DataLoader(validset, batch_size=batch_size)
print('Validation Set Loaded!')

model = models.resnet152(False) if not parser_args.debug else models.resnet18(False)
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)
writer = SummaryWriter(log_dir)


for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):
        t1 = time.time()
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        preds = model(images)
        loss = criterion(preds, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if (i+1) % 100 == 0:
            t2 = time.time()
            writer.add_scalar('Train Loss', loss.item(), (i+1)+(epoch)*total_step)
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Time: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item(), t2-t1))
            print('-'*80)
        
    
    if (epoch+1) % 10 == 0: #validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print('Accuracy of the model on the validation images: {} %'.format(100 * accuracy))
        writer.add_scalar('Valid Accuracy', accuracy, (i+1)+(epoch+1)*total_step)
        print('@'*80)
        model.train()
        torch.save(model.state_dict(), model_path+'model_{}.ckpt'.format(epoch)) 



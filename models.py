import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1   = nn.Linear(12*12*64, 1000)
        self.fc2   = nn.Linear(1000, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)
    
    # Forward pass
    def forward(self,x):
        '''
           input shape (Batch, Channel,Height,Width): (8,1,28,28)
           output shape (batch,121)
        
        '''
        #print(x.shape)
        
        # Conv1->relu->pool
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Conv2->pool1->relu->dropout
        x = self.dropout(self.pool1(F.relu(self.conv2(x))))
        # conv3
        
        x = self.conv3(x)
        
        # conv4->relu->pool
        x = self.pool2(F.relu(self.conv4(x)))
        
        # dropout
        x = self.dropout(x)
        
        #print(x.shape)
        #reshape
        x = x.view(-1,64*12*12)
        
        x = self.dropout(F.relu(self.fc1(x)))
        
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x
        
        
    
    

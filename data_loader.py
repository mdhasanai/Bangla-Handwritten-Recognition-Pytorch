import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision as tv
import pandas as pd
from matplotlib import image

from PIL import Image
import os
import pickle
import torchvision.transforms as transforms

from matplotlib import image
from PIL import Image
import torch
from numpy import asarray
import numpy as np
from skimage import io

def set_transform(horizontal_flip=False, normalize=True):
    compose_lst = []
    if horizontal_flip: compose_lst.append(transforms.RandomHorizontalFlip())
    compose_lst.append(transforms.ToTensor())
    if normalize: compose_lst.append(transforms.Normalize(mean=[0.456],
                             std=[0.229] ))

    transform = transforms.Compose(compose_lst)
    
    return transform
    
class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None,loader=tv.datasets.folder.default_loader):
        self.df = df
        self.transform=transform
        self.images= self.df['image']
        self.classes= self.df['ids']
        

    def __getitem__(self, index):
        img_id = self.images[index]
        
        #data = Image.open(img_id)#.reshape(1,28,28)
        image = io.imread(img_id)
        image = np.array(image).reshape(1,28,28)
        #print(image.shape)
        if self.transform is not None:
            image = self.transform(image)
            image = image.view(1,28,28)
        #print(image.shape)
        classes = self.classes[index]
        
        
        return image, classes


    def __len__(self):
        n, _ = self.df.shape
        return n
    

def get_loader( csv_file,batch_size=8, num_workers=4,transform=None,shuffle=False,):
 
    df = pd.read_csv(csv_file)
    train_dataset = ImagesDataset(df,transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle)

    return train_loader

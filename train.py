import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_loader import set_transform, get_loader
from models import Model
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from config import *



# Load the character and map
data = pd.read_csv("./data/labels.csv",index_col=False)
ids_to_char = {i : char for i,char in enumerate(data["chars"])}
print(f"Vocab size: {len(ids_to_char)}")

def train():

    # load data
    transform= set_transform()

    train_loader = get_loader(train_corpus,batch_size=8, transform=transform)
    valid_loader = get_loader(valid_corpus,batch_size=8, transform=transform)

    ## Define Model and print

    model = Model(vocab)
    print(model)
    
    
    # Defining Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    best_train_loss, best_valid_loss = 100000,100000
    train_loss, valid_loss = [],[]
    not_improved = 0
    show_after_iter = 10
    # Checking cuda is available or not

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        #print("Found GPU. Model Shifting to GPU")
        model.cuda()
        
    # for watching in tensorboard 
    tb = SummaryWriter()

    print("*"*30 + " Training Start "+ "*"*30 )
    for e in range(1, 2):
        
        ## Training Start ##
        model.train()
        for i,(image,classes) in enumerate(train_loader):

            if gpu_available:
                image = image.cuda()
                classes = classes.cuda()

            output = model(image)
            #_,pred = torch.max(output.data,1)

            loss = criterion(output,classes)

            # backprop
            loss.backward()
            optimizer.step()

            # loss move to cpu
            loss = loss.cpu().detach().numpy()
            train_loss.append(loss)
            
            if i % show_after_iter == 0:
                avg_loss = sum(train_loss)/len(train_loss)
                print(f"Epoch: ({e}/{epoch}) Loss: {loss} Avg Loss: {avg_loss} Accuracy: {100-loss} Avg Acc: {100-avg_loss}")

            if i == 3:
                break

        del image, loss, classes 
        
        avg_train_loss = sum(train_loss)/len(train_loss) 

        #print(f"Epoch: {e} Training Loss: {avg_train_loss} Training Accuracy: {100-avg_train_loss}")

        ## Validation Start ##
        model.eval()
        for i,(image,classes) in enumerate(valid_loader):

            if gpu_available:
                image = image.cuda()
                classes = classes.cuda()

            output = model(image)
            loss = criterion(output,classes)
            # loss move to cpu
            loss = loss.cpu().detach().numpy()
            valid_loss.append(loss)
            #print(f"Loss: {loss}")

            if i == 3:
                break
        avg_valid_loss = sum(valid_loss)/len(valid_loss)
        # save if model loss is improved
        if avg_valid_loss<best_valid_loss:
            best_train_loss = avg_valid_loss
            model_save = save_path+"/best_model.th"
            torch.save(model.state_dict(),model_save)
            not_improved = 0
        else:
            not_improved +=1

        if not_improved>=6:
            break
        
        print(f"\n\t Epoch: {e} Training Loss: {avg_train_loss} Training Accuracy: {100-avg_train_loss}")
        
        print(f"\t Epoch: {e} Validation Loss: {avg_valid_loss} Validation Accuracy: {100-avg_valid_loss} \n")
        
    # Saving training and validation losses so tha further graph can be generated
    save_loss = {"train":train_loss, "valid":valid_loss}
    with open(save_path+"/losses.pickle","wb") as files:
        pickle.dump(save_loss,files)

tb.close()

        
train()



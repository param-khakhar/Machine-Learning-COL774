from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import sys
import time

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

class Model(nn.Module):
    
    def __init__(self):

        super().__init__()
        self.hidden = nn.Linear(48 * 48, 100)
        self.output = nn.Linear(100,7)

    def forward(self,x):
        x = self.hidden(x)
        x = F.tanh(x)
        x = self.output(x)
        x = F.softmax(x)
        return x

if __name__ == "__main__":

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    out_file = sys.argv[3]

    X_df = pd.read_csv(train_file,header = None)
    X_df.head()

    ytrain_df = X_df[X_df.columns[0]]
    Xtrain_df = X_df[X_df.columns[1:]]

    ytrain = ytrain_df.values
    Xtrain = Xtrain_df.values

    X_df = pd.read_csv(test_file,header = None)
    ytest_df = X_df[X_df.columns[0]]
    Xtest_df = X_df[X_df.columns[1:]]
    ytest = ytest_df.values
    Xtest = Xtest_df.values

    model = Model()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    np.random.seed(7)
    count = 0
    epoch_loss = []
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(500):  # loop over the dataset multiple times

        running_loss = 0.0
        Xtrain, ytrain = shuffle(Xtrain,ytrain)
        batchesX = [Xtrain[100*i:100*i+100] for i in range(Xtrain.shape[0]//100)] 
        batchesy = [ytrain[100*i:100*i+100] for i in range(ytrain.shape[0]//100)]
        for x,y in zip(batchesX, batchesy):   
            inputs, labels = torch.tensor(x,dtype = torch.float), torch.tensor(y,dtype= torch.long)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
        
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        

        running_loss = running_loss/len(batchesX)
        if epoch == 0 :
            prev_loss = running_loss
            continue
        epoch_loss.append(running_loss)

        #if(abs(running_loss - prev_loss)<0.0000005):
        #  count += 1
        if len(epoch_loss) > 5:
            if np.max(epoch_loss[-5:]) - np.min(epoch_loss[-5:]) < 1e-4:
                break
        prev_loss = running_loss

    test = torch.tensor(Xtest,dtype = torch.float)
        #actual = torch.tensor(ytest,dtype= torch.long)
    test = test.to(device)
    pred = model(test)
    pred = torch.argmax(pred,axis=1)
    pred = pred.cpu()

    pred = pred.numpy()
    write_predictions(out_file,pred)


        #acc_test = torch.mean((pred == actual)*1.)
        #print("Test-Acc:",acc_test)

        #f1_test = f1_score(actual.cpu(),pred,average = 'macro')
        #print("F1-Test:",f1_test)

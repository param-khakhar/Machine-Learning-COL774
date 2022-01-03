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
    self.conv1 = nn.Conv2d(1,64,(3,3),3)
    self.conv2 = nn.Conv2d(64,128,(2,2),2)
    self.fc1 = nn.Linear(512,256)
    self.bn1 = nn.BatchNorm2d(64)
    self.bn2 = nn.BatchNorm2d(128)
    self.bn3 = nn.BatchNorm1d(256)
    self.fc2 = nn.Linear(256,7)
    self.bn4 = nn.BatchNorm1d(7)

  def forward(self,x):
    x = self.bn1(F.relu(self.conv1(x)))
    x = F.max_pool2d(x,(2,2),stride = 2)
    x = self.bn2(F.relu(self.conv2(x)))
    x = F.max_pool2d(x,(2,2),stride = 2)
    #fc1 = nn.Linear(self.num_flat_features(x),256)
    #print(fc1)
    x = x.view(-1,512)
    #print(x.shape)
    x = self.bn3(F.relu(self.fc1(x)))
    x = self.bn4(F.relu(self.fc2(x)))
    x = F.softmax(x)
    return x
  
  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

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
    for epoch in range(150):  # loop over the dataset multiple times

        running_loss = 0.0
        Xtrain, ytrain = shuffle(Xtrain,ytrain)
        batchesX = [Xtrain[100*i:100*i+100] for i in range(Xtrain.shape[0]//100)] 
        batchesy = [ytrain[100*i:100*i+100] for i in range(ytrain.shape[0]//100)]
        for x,y in zip(batchesX, batchesy):   
            x = x.reshape((x.shape[0],1,48,48))
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
        #prev_loss = running_loss

    test = torch.tensor(Xtest.reshape((Xtest.shape[0],1,48,48)),dtype = torch.float)
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

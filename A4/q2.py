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
    self.conv0 = nn.Conv2d(1,32,(3,3),1,padding=1)
    self.bn0 = nn.BatchNorm2d(32)
    self.conv1 = nn.Conv2d(32,64,(3,3),1,padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(64,64,(3,3),1,padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64,64,(3,3),1,padding=1)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64,64,(3,3),1,padding=1)
    self.bn4 = nn.BatchNorm2d(64)
    self.conv5 = nn.Conv2d(64,64,(3,3),1,padding=1)
    self.bn5 = nn.BatchNorm2d(64)
    self.conv6 = nn.Conv2d(64,64,(3,3),1,padding=1)
    self.bn6 = nn.BatchNorm2d(64)
    self.conv7 = nn.Conv2d(64,128,(3,3),1,padding=1)
    self.bn7 = nn.BatchNorm2d(128)

    self.fc1 = nn.Linear(128*3*3,512)
    self.bn8 = nn.BatchNorm1d(512)
    self.fc2 = nn.Linear(512,128)
    self.bn9 = nn.BatchNorm1d(128)
    self.fc3 = nn.Linear(128,7)
    self.bn10 = nn.BatchNorm1d(7)

    self.drop1d = nn.Dropout(0.5)

  def forward(self,x):
    
    x = self.bn0(F.relu(self.conv0(x)))
    x = self.bn1(F.relu(self.conv1(x)))
    x = F.max_pool2d(x,(2,2),stride = 2)
    #print(x.shape)
    x = self.bn2(F.relu(self.conv2(x)))
    x = self.bn3(F.relu(self.conv3(x)))
    x = F.max_pool2d(x,(2,2),stride = 2)

    x = self.bn4(F.relu(self.conv4(x)))
    x = self.bn5(F.relu(self.conv5(x)))
    x = F.max_pool2d(x,(2,2),stride = 2)

    x = self.bn6(F.relu(self.conv6(x)))
    x = self.bn7(F.relu(self.conv7(x)))
    x = F.max_pool2d(x,(2,2),stride = 2)

    #print(x.shape)
    x = x.view(-1,128*3*3)
    
    x = self.bn8(F.relu(self.fc1(x)))
    x = self.bn9(F.relu(self.fc2(x)))
    x = self.bn10(F.relu(self.fc3(x)))
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
    ytestp = ytest_df.values
    Xtestp = Xtest_df.values

    model = Model()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    val_scores = []
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    np.random.seed(7)
    count = 0
    epoch_loss = []
    optimizer = optim.Adam(model.parameters(), lr = 0.1)

    batch_size = 100

    for epoch in range(3000):  # loop over the dataset multiple times

        running_loss = 0.0
        Xtrain, ytrain = shuffle(Xtrain,ytrain)
        batchesX = [Xtrain[batch_size*i:batch_size*i+batch_size] for i in range(Xtrain.shape[0]//batch_size)] 
        batchesy = [ytrain[batch_size*i:batch_size*i+batch_size] for i in range(ytrain.shape[0]//batch_size)]
        
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
        # if (epoch % 10) == 0 :
        #     print('Epochs:',epoch,'Loss:',running_loss)

        #     dataloader = torch.utils.data.DataLoader(test, batch_size=1000, shuffle=False)

        #     right = []
        #     for i, left in enumerate(dataloader):
        #         #print(i,left.shape)
        #         with torch.no_grad():
        #             temp = model(left)
                
        #         temp = torch.argmax(temp,axis = 1)
        #         right = right + list(temp.to('cpu'))
        #         #right.append(temp.to('cpu'))
        #         del temp
        #         torch.cuda.empty_cache()

        # acc_test = torch.mean((pred == actual)*1.)
        # print("Test-Acc:",acc_test)

        # f1_test = f1_score(actual.cpu(),pred,average = 'macro')
        # print("F1-Test:",f1_test)
        # print()
        
        if epoch == 0 :
            prev_loss = running_loss
            continue

        epoch_loss.append(running_loss)

        #if(abs(running_loss - prev_loss)<0.0000005):
        #  count += 1
        # if len(epoch_loss) > 5:
        #   if np.max(epoch_loss[-5:]) - np.min(epoch_loss[-5:]) < 1e-4:
        #     break
        #prev_loss = running_loss

    #print('Loss is: ',running_loss)
    #print('Finished Training')

    test = torch.tensor(Xtestp.reshape(Xtestp.shape[0],1,48,48),dtype = torch.float)
    actual = torch.tensor(ytestp,dtype= torch.long)
    test = test.to(device)
    dataloader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False)

    right = []
    for i, left in enumerate(dataloader):
        #print(i,left.shape)
        with torch.no_grad():
            temp = model(left)
        
        temp = torch.argmax(temp,axis = 1)
        right = right + list(temp.to('cpu'))
        #right.append(temp.to('cpu'))
        del temp
        torch.cuda.empty_cache()

    #pred = model(test)
    #pred = torch.argmax(pred,axis=1)
    #actual = actual.to(device)
    #print(actual.shape,pred.shape)
    #pred = pred.cpu()
    pred = torch.tensor(right, dtype = torch.long)
    #acc_testp = torch.mean((pred == actual)*1.)
    #print("Private-Test-Acc:",acc_testp)

    #f1_test = f1_score(actual.cpu(),pred.cpu(),average = 'macro')
    #print("F1-Private-Test:",f1_test)

    #df_save = pd.DataFrame(columns = ["Id","Prediction"])
    #df_save["Id"] = [i for i in range(1,pred.numpy().shape[0]+1)]
    #df_save["Prediction"] = pred.numpy()
    #df_save.to_csv('submit.csv',index = None)
    #print(df_save.shape)

    pred = pred.numpy()
    write_predictions(out_file,pred)

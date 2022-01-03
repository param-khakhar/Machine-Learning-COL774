import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import numpy as np
import sys
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

class Model3(nn.Module):
  
  def __init__(self):

    super().__init__()
    self.hidden = nn.Linear(48*48, 100)
    self.output = nn.Linear(100,7)

  def forward(self,x):
    x = self.hidden(x)
    x = F.relu(x)
    x = self.output(x)
    x = F.softmax(x)
    return x

if __name__ == "__main__":

  X_df = pd.read_csv(sys.argv[1],header = None)
  X_df.head()

  ytrain_df = X_df[X_df.columns[0]]
  Xtrain_df = X_df[X_df.columns[1:]]

  ytrain = ytrain_df.values
  Xtrain = Xtrain_df.values

  X_df = pd.read_csv(sys.argv[2],header = None)
  ytest_df = X_df[X_df.columns[0]]
  Xtest_df = X_df[X_df.columns[1:]]
  ytest = ytest_df.values
  Xtest = Xtest_df.values

  temp = np.array(Xtrain)
  convert_train = []
  for i in range(temp.shape[0]):
    fd, hog_image = hog(temp[i].reshape(48,48), orientations=8, pixels_per_cell=(16, 16),cells_per_block=(3, 3), visualize=True)
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    convert_train.append(hog_image_rescaled)

  convert_train = np.array(convert_train)

  temp = np.array(Xtest)
  convert_test = []
  for i in range(temp.shape[0]):
    fd, hog_image = hog(temp[i].reshape(48,48), orientations=8, pixels_per_cell=(16, 16),cells_per_block=(3,3), visualize=True)
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    convert_test.append(hog_image_rescaled)

  convert_test = np.array(convert_test)

  model3 = Model3()
  if torch.cuda.is_available():
        device = torch.device('cuda:0')
  else:
        device = torch.device('cpu')

  model3 = model3.to(device)
  batchesX = [convert_train[100*i:100*i+100] for i in range(convert_train.shape[0]//100)] 
  batchesy = [ytrain[100*i:100*i+100] for i in range(ytrain.shape[0]//100)]
  criterion = nn.CrossEntropyLoss()
  epoch=0
  prev_loss = 0.00
  while(True):  # loop over the dataset multiple times
      running_loss = 0.0
      epoch = epoch + 1
      for x,y in zip(batchesX, batchesy):   
          inputs, labels = torch.tensor(x.reshape(-1,2304),dtype = torch.float), torch.tensor(y,dtype= torch.long)
          inputs = inputs.to(device)
          labels = labels.to(device)
          optimizer = optim.SGD(model3.parameters(), lr=0.1)
          # zero the parameter gradients
          optimizer.zero_grad()
          outputs = model3(inputs)
          loss = criterion(outputs,labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
      
      running_loss = running_loss/len(batchesX)
      
      if epoch == 0 :
        prev_loss = running_loss
        continue

      if(abs(running_loss - prev_loss)<0.00001 or epoch > 1000):
        break
      prev_loss = running_loss


  test = torch.tensor(convert_test.reshape(Xtest.shape[0],48*48),dtype = torch.float)
  test = test.to(device)
  actual = torch.tensor(ytest,dtype= torch.long)
  dataloader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False)

  right = []
  for i, left in enumerate(dataloader):
      #print(i,left.shape)
      with torch.no_grad():
          temp = model3(left)    
      temp = torch.argmax(temp,axis = 1)
      right = right + list(temp)
      #right.append(temp.to('cpu'))
      del temp
      
  pred = torch.tensor(right, dtype = torch.long)
  pred = pred.numpy()

  write_predictions(sys.argv[3],pred)
  # df_save = pd.DataFrame(columns = ["Id","Prediction"])
  # df_save["Id"] = [i for i in range(1,pred.numpy().shape[0]+1)]
  # df_save["Prediction"] = pred.numpy()
  # df_save.to_csv(sys.argv[3],index = None)

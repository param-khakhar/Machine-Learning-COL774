import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import sys

# Generate Dataset:

def sample():
    '''
    Using functions from numpy to sample values from normal distribution.
    '''
    theta_parameters = np.array([[3],[1],[2]])
    x1 = np.random.normal(3,2,size = [1000000,1])
    x2 = np.random.normal(-1,2,size = [1000000,1])

    X = np.ones([1000000,3])
    X[:,1:2] = X[:,1:2] * x1
    X[:,2:] = X[:,2:] * x2

    Y =  np.matmul(X,theta_parameters)
    Y = Y + np.random.normal(0,np.sqrt(2),size = Y.shape)

    return X,Y

if __name__ == "__main__":
    sample()
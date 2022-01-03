import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import sys
from q4a import LDA
import pandas as pd


if __name__ == "__main__":
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    X0 = np.loadtxt(data_dir + '/q4x.dat',unpack = True,dtype = float,usecols=[0])
    X1 = np.loadtxt(data_dir + '/q4x.dat',unpack = True,dtype = float,usecols=[1])
    Y = np.loadtxt(data_dir + '/q4y.dat',unpack = True,dtype = str)

    X = np.zeros([X0.shape[0],2])
    X[:,0] = X[:,0] + X0
    X[:,1] = X[:,1] + X1

    Y = (Y == 'Alaska')*1

    X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

    fig,ax = plt.subplots()
    y = Y
    x = X
    mask = (y == 1).reshape([y.shape[0]])
    ones = x[mask]
    mask = (y == 0).reshape([y.shape[0]])
    zeros = x[mask]
    ax.set_xlim((-4,4))
    ax.set_ylim((-4,4))
    ax.set_title("q4b: Training Data")
    ax.set_xlabel("")
    ax.scatter(ones[:,0],ones[:,1],color = 'r',label = 'Alaska')
    ax.scatter(zeros[:,0],zeros[:,1],label = 'Canada')
    ax.legend(loc = 'best')
    ax.set_xlabel("Feature - 1")
    ax.set_ylabel("Feature - 2")
    fig.savefig(output_dir+'/q4b.png',dpi = 200)
    plt.close("all")


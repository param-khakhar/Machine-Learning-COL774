import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")
import sys
from q4a import LDA
from q4d import QDA
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


    y = Y
    x = X

    model = QDA()
    phi,mu0,mu1,sigma0,sigma1 = model.fit(X,Y)

    fig,ax = plt.subplots()
    x1 = np.linspace(-4,4,500)
    x2 = np.linspace(-4,4,500)
    xx1,xx2 = np.meshgrid(x1,x2)
    xx = np.zeros((x1.shape[0]*x2.shape[0],2))
    xx[:,0] = xx1.reshape(-1)
    xx[:,1] = xx2.reshape(-1)

    #print(sigma0,sigma1)

    # Separating the points of the plane on the basis of their class.
    preds = model.pred_class(xx,mu0,mu1,sigma0,sigma1)
    ax.contour(x1,x2,preds.reshape((x1.shape[0],x2.shape[0])),alpha=1,colors = 'purple',linewidths = 0.5)

    mask = (y == 1).reshape([y.shape[0]])
    ones = x[mask]
    mask = (y == 0).reshape([y.shape[0]])
    zeros = x[mask]
    ax.set_xlim((-4,4))
    ax.set_ylim((-4,4))
    ax.set_title("q4e: QDA vs LDA")
    ax.set_xlabel("")
    ax.scatter(ones[:,0],ones[:,1],color = 'r',label = 'Alaska')
    ax.scatter(zeros[:,0],zeros[:,1],label = 'Canada')
    #ax.legend(loc = 'best')
    ax.set_xlabel("Feature - 1")
    ax.set_ylabel("Feature - 2")

    model = LDA()
    phi,mu0,mu1,sigma =  model.fit(X,Y)

    linex = np.array(ax.get_xlim())
    temp = np.matmul(np.linalg.inv(sigma),(mu0-mu1).T)
    liney = (np.matmul((mu0 + mu1),temp) - temp[0]*linex)/temp[1]
    ax.plot(linex,liney,label = 'Decision Boundary',color = 'black')

    purple_patch = mpatches.Patch(color='purple', label='Quadratic Boundary')
    black_patch = mpatches.Patch(color='black', label='Linear Boudary')
    ax.legend(handles = [purple_patch,black_patch],loc = 'best')

    #    plt.show()
    fig.savefig(output_dir+'/q4e.png',dpi = 200)
    plt.close("all")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import sys
from q3a import LogisticRegression
import pandas as pd

if __name__ == "__main__":

    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    X = pd.read_csv(data_dir+"/logisticX.csv")
    Y = pd.read_csv(data_dir+"/logisticY.csv")
    xo = X.values
    X = (X - X.mean()) / X.std()
    x  = X.values

    mu1 = np.mean(x[:,0])
    sigma1 = np.std(x[:,0])
    mu2 = np.mean(x[:,1])
    sigma2 = np.std(x[:,1])

    newX = np.ones([X.shape[0],X.shape[1]+1])
    newX[:,1:] = newX[:,1:]*X   
    model = LogisticRegression(newX.shape[1])
    params = model.fit(newX,Y.values)
    #print(params)

    # Plotting a scatter plot followed by points on a straight line.
    fig,ax = plt.subplots()
    y = Y.values
    mask = (y == 1).reshape([y.shape[0]])
    ones = xo[mask]
    mask = (y == 0).reshape([y.shape[0]])
    zeros = xo[mask]
    #ax.set_xlim((-4,4))
    #ax.set_ylim((-4,4))
    ax.set_title("q3b: Logisitic Regression - Decision Boundary")
    ax.set_xlabel("")
    ax.scatter(ones[:,0],ones[:,1],color = 'r',label = 'Label - 1')
    ax.scatter(zeros[:,0],zeros[:,1],label = 'Label - 0')

    linex = np.array(ax.get_xlim())
    liney = mu2+(-sigma2/params[2])*(params[1]*linex/sigma1 - params[1]*mu1/sigma1 + params[0])
    ax.plot(linex,liney,label = 'Decision Boundary',color = 'black')
    ax.legend(loc = 'best')
    fig.savefig(output_dir+'/q3b.png',dpi = 200)
    plt.close('all')



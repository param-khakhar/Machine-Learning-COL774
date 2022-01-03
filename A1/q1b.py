from q1a import LinearRegression

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import sys
from matplotlib import pyplot as plt



if __name__ == "__main__":
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    X = pd.read_csv(data_dir+"/linearX.csv")
    Y = pd.read_csv(data_dir+"/linearY.csv")
    mu = np.mean(X.values)
    sigma = np.std(X.values)
    Xo = X
    X = (X - X.mean())/X.std()
    #print(Y.shape,X.shape)
    newX = np.ones([X.shape[0],X.shape[1]+1])
    newX[:,1:] = newX[:,1:]*X   
    model = LinearRegression(newX.shape[1],0.025)
    params = model.fit(newX,Y.values)   
    ypred = model.predict(newX)
    ypred = ypred.reshape(-1)
    
    #Using Matplotlib for making graphs.
    fig,ax = plt.subplots()
    ax.scatter(X.values,Y.values,color = 'r',label = "Original")
    ax.set_title("Q1b")
    ax.set_xlabel("Acidity of the Wine")
    ax.set_ylabel("Density of the Wine")
    ax.plot(X.values,ypred,label = 'Hypothesis')
    ax.legend(loc = 'best')
    fig.savefig(output_dir+'/q1b.png',dpi = 125)

    #print("Complete")

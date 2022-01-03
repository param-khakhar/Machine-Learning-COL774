import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import time
matplotlib.use("Agg")
import sys
from q2a import sample
from q2b import SGD


if __name__ == "__main__":
    output_dir = sys.argv[2]
    data_dir = sys.argv[1]

    df = pd.read_csv(data_dir+"/q2test.csv")
    Xtest = df[df.columns[:-1]]
    Ytest = df[df.columns[-1]]

    # Gettting the X's and Y's from the function implemented in the file q2a.py.
    X,Y = sample()

    # Concatenating the two arrays in a left/right fashion.
    combined = np.hstack((X,Y))

    # Shuffling the rows of the combined array, and separating the X's and Y's.
    np.random.shuffle(combined)

    X = combined[:,:-1]
    Y = combined[:,-1:]

    Xtest = Xtest.values
    Xtest_ = np.ones([Xtest.shape[0],Xtest.shape[1]+1])
    Xtest_[:,1:] = Xtest_[:,1:] * Xtest

    batches = [1,100,10000,1000000]

    # Computing the error from the original parameters.
    original = np.array([[3.0],[1.0],[2.0]])
    Ypred_original = np.matmul(Xtest_,original)
    Ypred_original = Ypred_original.reshape(Ypred_original.shape[0])

    with open(output_dir+'/q2c.txt', 'w') as f:
        #print(params)
        print("Error:",np.mean((Ytest.values - Ypred_original)*(Ytest.values-Ypred_original))/2,file = f)
    
    # Computing the error for different learned parameters for different batch sizes.
        for batch in batches:
            model = SGD(X.shape[1],batch)
            params,Jthetalist,Thetalist = model.fit(X,Y)
            Ypred = model.predict(Xtest_)
            Ypred = Ypred.reshape(Ypred.shape[0])
            print("Error_"+str(batch)+":",np.mean((Ytest.values - Ypred)*(Ytest.values-Ypred))/2,file = f)

    f.close()
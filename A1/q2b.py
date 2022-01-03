import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import time
matplotlib.use("Agg")
import sys
from q2a import sample

def cost_function(X,Y,B):
    predictions = np.dot(X,B)
    cost = (1/len(Y)) * np.sum((predictions - Y) ** 2)
    return cost

class SGD:

    def __init__(self, n, b):
        self.theta_parameter = np.zeros([n,1])
        self.batch_size = b

    def fit(self,X,y):

        start = time.time()

        updates = 0
        converged = 0
        lr = 0.001
        t = 0
        epochs = 0
        m = X.shape[0]
        theta = self.theta_parameter

        Jlist = []
        Thetalist = [[] for i in range(X.shape[1])]

        batchesX = []
        batchesY = []
        b = self.batch_size

        temp = []

        for i in range(m//b):
            batchesX.append(X[b*i:(b*i+b),:])
            batchesY.append(y[b*i:(b*i+b),:])
            htheta = np.matmul(batchesX[i],theta)
            
        temp = []
        while True:
            cost_batches = []
            for l in range(m//b):
                htheta = np.matmul(batchesX[l],theta)
                delJtheta = np.matmul(batchesX[l].T, (batchesY[l]-htheta))/b
                theta = theta + (lr * delJtheta)
                Jtheta = np.matmul((batchesY[l]-htheta).T,(batchesY[l]-htheta))/(2*b)
                cost_batches.append(Jtheta[0][0])

                for i in range(theta.shape[0]):
                    Thetalist[i].append(theta[i,:][0])

                if len(temp) > 0 and abs(cost_batches[l] - temp[l]) < 1e-8:
                    converged += 1
                t += 1

            if converged >= (m // b):
                self.theta_parameter = theta
                break
            epochs += 1
            temp = cost_batches
            Jlist.append(sum(temp))
            
        #print(time.time() - start, "Time:")
        return theta,Jlist,Thetalist

    def predict(self,X):
        return np.matmul(X,self.theta_parameter)


if __name__ == "__main__":
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    # Gettting the X's and Y's from the function implemented in the file q2a.py.
    X,Y = sample()

    # Concatenating the two arrays in a left/right fashion.
    combined = np.hstack((X,Y))

    # Shuffling the rows of the combined array, and separating the X's and Y's.
    np.random.shuffle(combined)

    X = combined[:,:-1]
    Y = combined[:,-1:]
    with open(output_dir+'/q2b.txt', 'w') as f:
        batches = [1,100,10000,1000000]
        for batch in batches:
            model = SGD(X.shape[1],batch)

            params,Jthetalist,Thetalist = model.fit(X,Y)
                #print(params)
            print('Batch Size:',batch,'\nTheta0:',params[0][0],'\nTheta1:',params[1][0],'\nTheta2:',params[2][0], '\n',file=f)
        f.close()

'''
b = 1e6

Threshold: 1e-08 Epochs: 20284 Theta: [2.989,1.002,1.999] Time: 523s

b = 1e4

Threshold: 1e-08 Epochs: 277 Theta: [2.997,1.000,2.000] Time: 5.51s

b = 1e2

Threshold: 1e-08 Epochs: 7 epochs Theta: [3.000,0.9954,2.0013] Time: 1.23s

b = 1

Threshold: 1e-08 Epochs: 2 epochs Theta: [3.007,0.9896,1.9933] Time: 38.4s

'''
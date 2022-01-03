import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import sys

def sigmoid(arr):
    return 1/(1 + np.exp(-arr)) 

def cost(X,Y,s):
    return 

class LogisticRegression:

    def __init__(self, n):
        self.theta_parameter = np.zeros(n)

    def fit(self,X,y):
        '''
        Function for computing the optimal values of theta-parameters using Newton's Method.
        '''

        theta = self.theta_parameter
        htheta = sigmoid(X.dot(theta))
        temp = (y*np.log(htheta) + (1-y)*(1-np.log(htheta)))/y.shape[0]
        y = y.reshape([y.shape[0]])
        steps = 0

        while True:
            # Computing gradient, hessian and then inverse for the update step, in vectorized form.
            grad =  X.T.dot(y-htheta)
            temp1 = htheta * (1-htheta)
            w = np.diag(temp1)
            hess = np.matmul(np.matmul(X.T,w),X)
            try:
                theta = theta + np.linalg.inv(hess).dot(grad)
            except np.linalg.LinAlgError:
                #print("Steps:",steps)
                break

            htheta = sigmoid(X.dot(theta))
            Jtheta = (y*np.log(htheta) + (1-y)*(1-np.log(htheta)))/y.shape[0]

            if abs(np.sum(temp)-np.sum(Jtheta)) < 1e-8:
                self.theta_parameter = theta
                #print("Steps:",steps)
                #print("Thresh:",abs(np.sum(temp)-np.sum(Jtheta)))
                return theta
                break
            steps += 1
            temp = Jtheta

    def predict(self,X):
        return np.matmul(X,self.theta_parameter)

if __name__ == "__main__":
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    X = pd.read_csv(data_dir+"/logisticX.csv")
    Y = pd.read_csv(data_dir+"/logisticY.csv")
    X = (X - X.mean()) / X.std()
    #print(Y.shape,X.shape)

    newX = np.ones([X.shape[0],X.shape[1]+1])
    newX[:,1:] = newX[:,1:]*X   
    model = LogisticRegression(newX.shape[1])

    params = model.fit(newX,Y.values)

    with open(output_dir+'/q3a.txt', 'w') as f:
        print("Theta0:",params[0],'\nTheta1:',params[1],'\nTheta2:',params[2],file = f)
    f.close()






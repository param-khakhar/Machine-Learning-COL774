import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import sys

class LinearRegression:

    def __init__(self, n, lr):
        self.theta_parameter = np.zeros([n,1])
        self.eta = lr

    def fit(self,X,y):
        '''
        Method for invoking the training of the model, using the features
        and the values of the target variable. Also returns the cost at
        the end of each iteration in a list Jlist, and the values of Theta
        Parameter at the end of each iteration, in order to plot graphs.
        '''
        Jlist = []
        Thetalist = [[] for i in range(X.shape[1])]
        lr = self.eta
        converged = False
        t = 0
        m = X.shape[0]
        theta = self.theta_parameter
        htheta = np.matmul(X,theta)

        temp = np.matmul((y-htheta).T,(y-htheta))/(2*m)

        while True:

            delJtheta = np.matmul(X.T, (y-htheta))/m    #Vectorized implementation for calculating the gradients.
            theta = theta + lr * delJtheta
            htheta = np.matmul(X,theta)
            Jtheta = np.matmul((y-htheta).T,(y-htheta))/(2*m)
            diff = abs((Jtheta-temp)[0][0])

            if diff <= 1e-10:
                #print(t,"Iterations")
                #print("MSE:",Jtheta)  
                self.theta_parameter = theta    #Saving the values to the theta_parameter in the object.
                break
            temp = Jtheta
            t+=1
            Jlist.append(temp[0][0])
            for i in range(theta.shape[0]):
                Thetalist[i].append(theta[i,:][0])
        return theta,Jlist,Thetalist

    def predict(self,X):
        return np.matmul(X,self.theta_parameter)    #Make predictions.


if __name__ == "__main__":

    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Reading the csv files using pandas.
    X = pd.read_csv(data_dir+"/linearX.csv")
    Y = pd.read_csv(data_dir+"/linearY.csv")

    # Standardizing the dataframe X, to have zero mean and unit variance for all the respective features.
    X = (X - X.mean()) / X.std()

    # Incorporating the Intercept Term.
    newX = np.ones([X.shape[0],X.shape[1]+1])
    newX[:,1:] = newX[:,1:]*X

    #Initializing the model object
    model = LinearRegression(newX.shape[1], 0.025)
    params,Jthetalist,Thetalist = model.fit(newX,Y.values)
    
    with open(output_dir+'/q1a.txt', 'w') as f:
        print("Theta0:",params[0][0],"\nTheta1:",params[1][0], file=f)
    f.close()






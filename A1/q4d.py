import numpy as np
import sys

class QDA:
    def __init__(self, s = True):
        self.same = s

    def fit(self,X,Y):
        phi = np.mean(Y)
        mu0 = np.sum(X[Y == 0],axis = 0)/np.sum(Y==0)
        mu1 = np.sum(X[Y == 1],axis = 0)/np.sum(Y == 1)
        mu = [mu0,mu1]
        temp0 = X[Y == 0]
        temp1 = X[Y == 1]
        temp0 -= mu0
        temp1 -= mu1

        sigma0 = np.matmul(temp0.T,temp0)
        sigma0 /= np.sum(Y == 0)

        sigma1 = np.matmul(temp1.T,temp1)
        sigma1 /= np.sum(Y == 1)

        # print(sigma0,sigma1)

        # sigma0 = np.cov(np.transpose(temp0))
        # sigma1 = np.cov(np.transpose(temp1))
        return phi, mu0, mu1, sigma0, sigma1
    
    def pred_class(self,X,mu0,mu1,sigma0,sigma1):
        a = 2
        ypred = []
        for i in range(X.shape[0]):

            b0 = X[i, :] - mu0
            temp0 = (-1/2)*np.dot(np.dot(b0.T, np.linalg.inv(sigma0)), b0)
            g = 1 / np.sqrt((2*np.pi**a)*np.linalg.det(sigma0))
            score0 = g * np.e**temp0 

            b1 = X[i, :] - mu1
            temp1 = (-1/2)*np.dot(np.dot(b1.T, np.linalg.inv(sigma1)), b1)
            g = 1 / np.sqrt((2*np.pi**a)*np.linalg.det(sigma1))
            score1 = g * np.e**temp1 

            if score0 > score1:
                ypred.append(0)
            else:
                ypred.append(1)

        return np.array(ypred)

if __name__ == "__main__":
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    X0 = np.loadtxt(data_dir + '/q4x.dat',unpack = True,dtype = float,usecols=[0])
    X1 = np.loadtxt(data_dir + '/q4x.dat',unpack = True,dtype = float,usecols=[1])
    Y = np.loadtxt(data_dir + '/q4y.dat',unpack = True,dtype = str)

    X = np.zeros([X0.shape[0],2])
    X[:,0] = X[:,0] + X0
    X[:,1] = X[:,1] + X1

    # X = X.reshape([X.shape[1],X.shape[0]])
    Y = (Y == 'Alaska')*1

    #print(X.mean())

    #print(X,Y)

    X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

    X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

    model = QDA()
    phi, mu0, mu1, sigma0, sigma1 = model.fit(X,Y)

    with open(output_dir+'/q4d.txt', 'w') as f:
        print("Phi:",phi,file = f)
        print("Mu0:",mu0, file = f)
        print("Mu1:",mu1, file = f)
        print("Sigma0:",sigma0, file = f)
        print("Sigma1:",sigma1, file = f)
    f.close()



import numpy as np
import sys

class LDA:
    def __init__(self, s = True):
        self.same = s

    def fit(self,X,Y):
        phi = np.mean(Y)
        mu0 = np.sum(X[Y == 0],axis = 0)/np.sum(Y==0)
        mu1 = np.sum(X[Y == 1],axis = 0)/np.sum(Y == 1)
        #print("M",mu1)
        conf = np.zeros(2)
        for i in range(X.shape[0]):
            if Y[i] == 1:
                conf = conf + X[i]
        conf = conf/np.sum(Y == 0)
        mu = [mu0,mu1]
        temp = X
        temp[Y == 0] -= mu0
        temp[Y == 1] -= mu1
        sigma = np.matmul(temp.T,temp)
        sigma/=X.shape[0]
        return phi, mu0, mu1, sigma


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

    model = LDA()
    phi, mu0, mu1, sigma = model.fit(X,Y)

    with open(output_dir+'/q4a.txt', 'w') as f:
        print("Phi:",phi,file = f)
        print("Mu0:",mu0,file = f)
        print("Mu1:",mu1, file = f)
        print("Sigma:",sigma, file = f)
    f.close()



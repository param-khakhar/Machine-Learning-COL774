import pandas as pd
import numpy as np
import sys
import cvxopt
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

def compute_score(X,y,classifiers,classes = False):

    '''
    Function which predicts the labels for the points in X, given all the classifiers
    and the true_labels y. Votes are assigned to all the classes based on the predictions
    by the classifiers, and if the top 2 classes have the same number of votes, the assigned
    class would be the one which as a higher value of |w.x + b|. The value |w.x + b| is also,
    maintained while classifying the instances.
    '''

    ypred = np.zeros((10,X.shape[0]))
    ypred_score = np.zeros((10,X.shape[0]))

    start = time.time()
    for k in classifiers.keys():
        classifiers[k].gamma = 0.05
        label = classifiers[k].predict(X)
        scores = classifiers[k].predict_score(X)
        first = (1*(label == -1)).reshape(-1)
        second = (1*(label == 1)).reshape(-1)
        ypred[k[0]] += first
        ypred[k[1]] += second
        scores = scores.reshape(-1)
        ypred_score[k[0]] += scores
        ypred_score[k[1]] += scores

    ypreds = np.zeros(X.shape[0])

    for i in range(ypred.shape[1]):

        temp = ypred[:,i].copy()
        temp.sort()
        if temp[-1] != temp[-2]:
            ypreds[i] = np.argmax(ypred[:,i])
        else:
            j = 8
            while j >= 0 and temp[j] == temp[j+1]:
                j -= 1
            ypreds[i] = np.argmax(np.abs(ypred_score[j+1:,i]))

    if classes == True:
        #print(np.mean(y == ypreds))
        return ypreds
    return np.mean(y == ypreds)

def form_kernel(X,Y,gamma):

    '''
    Given two matrices X and Y of dimenions (m,n) and (m',n) respectively, 
    this function returs a matrix R of dimension (m,m'), such that 
    R[i][j] = exp(-gamma * || X[i]-Y[j] || ^ 2)
    '''

    Z = np.linalg.norm(X,axis = 1) ** 2
    Z = Z.reshape(-1,1)
    temp1 = np.ones((X.shape[0],Y.shape[0])) * Z
    Z = np.linalg.norm(Y,axis = 1) ** 2
    Z = Z.reshape(1,-1)
    temp2 = np.ones((X.shape[0],Y.shape[0])) * Z
    temp3 = np.matmul(X,Y.T)

    return np.exp(-gamma * (temp1 + temp2 + -2*temp3))

def process_data(df, label1, label2):

    '''Function for processing the dataframes, and returns the corresponding
       numpy arrays for the labels in the arguments.
    '''

    train = df[(df[df.columns[-1]] == label1) | (df[df.columns[-1]] == label2)]
    Xtrain = train[train.columns[:-1]].values
    ytrain = train[train.columns[-1]].values
    Xtrain = Xtrain/255
    ytrain = -1 * (ytrain == label1) + 1 * (ytrain == label2)

    return Xtrain,ytrain

# Class SVM contains the parameters and the methods used for fitting and making predictions.

class SVM:
    def __init__(self,kernel = "Linear",gamma = 0, c = 1, classes = 2):
        self.w = None
        self.b = None
        self.k = kernel
        self.gamma = gamma
        self.classes = 2
        self.alpha = None
        self.c = c
        self.y_alpha = None         # Vector whose ith index is alpha_i * y_i
        self.Xt = None              # Training data, to be stored in case of Gaussian Kernel

    def fit(self,Xtrain,ytrain):

        '''
        Computes the parameters P, q, G, h, A, b for CVXOPT and solves to get the alpha_i. If the kernel is "Linear", then calculate the weight matrix, and store it. 
        If the kernel is "Gaussian", need to store the training data. The bias parameter is calculated as well.
        '''
        ytrain = ytrain.reshape(-1,1) * 1.
        m = Xtrain.shape[0]

        if self.k == "Linear":

            P = Xtrain * ytrain
            P = np.dot(P,P.T)
            q = -np.ones((m,1))
            G = np.vstack((np.eye(m)*-1,np.eye(m)))
            h = np.vstack((np.zeros((m,1)),np.ones((m,1)) * self.c))
            A = ytrain.reshape(1,-1) * 1.
            b = np.zeros(1) * 1.
            b = b.reshape(-1,1)

        else:

            Z = np.linalg.norm(Xtrain,axis = 1) ** 2
            Z = Z.reshape(-1,1)
            temp = np.ones((m,m)) * Z
            K = temp + -2 * np.matmul(Xtrain,Xtrain.T) + temp.T
            K = np.exp(-0.05 * K)
            P = np.matmul(ytrain,ytrain.T) * K
            
            q = -np.ones((m,1))
            G = np.vstack((np.eye(m)*-1,np.eye(m)))
            h = np.vstack((np.zeros((m,1)),np.ones((m,1)) * self.c * 1.))
            A = ytrain.reshape(1,-1) * 1.
            b = np.zeros(1) * 1.
            b = b.reshape(-1,1)

        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)

        sol = cvxopt.solvers.qp(P,q,G,h,A,b)
        self.alpha = np.array(sol['x'])
        
        if self.k == "Linear":

            self.w = np.sum(self.alpha * (Xtrain * ytrain),axis = 0)
            negs = (ytrain == -1).reshape(-1)
            pos = (ytrain == 1).reshape(-1)
            ones = np.matmul(Xtrain[pos],self.w.T)
            m_ones = np.matmul(Xtrain[negs],self.w.T)
            ones = ones.reshape(-1)
            m_ones = m_ones.reshape(-1)
            mn = np.min(ones)
            mx = np.max(m_ones)
            self.b = -(mn+mx)/2

        else:

            svs = (self.alpha > 1e-5) & (self.alpha < self.c)
            svs = svs.reshape(-1)
            temp = ytrain.reshape(-1,1)
            y_alpha = temp * self.alpha
            y_alpha = y_alpha.reshape(1,-1)
            self.y_alpha = y_alpha
            temp = ytrain[svs]
            temp = temp.reshape(-1)
            self.b = np.mean(temp - np.sum((y_alpha * K)[svs],axis = 1))
            self.Xt = Xtrain


    def score(self,X,y):

        '''
        Given the test data X, and the test (true) labels y, the function returns the
        accuracy of the predictions made by the model.
        '''

        if self.k == "Linear":

            temp = (np.matmul(X,self.w.T) + self.b)
            ypred_val = (temp > 0) * 1 + (temp < 0) * -1
            return np.mean(ypred_val == y)

        else:
            
            ypred_val = np.ones((X.shape[0],1))
            y_alpha = self.y_alpha.reshape(1,-1)
        
            kernel = form_kernel(self.Xt,X,self.gamma)
            wtphi = np.matmul(y_alpha,kernel)
            wtphi += self.b
            
            ypred_val = (wtphi > 0) * 1 + (wtphi < 0) * -1
            return np.mean(ypred_val == y)

    def predict(self,X):

        "This function returns the predictions (labels) for the corresponding training points X."

        if self.k == "Linear":
            temp = (np.matmul(X,self.w.T) + self.b)
            ypred_val = (temp > 0) * 1 + (temp < 0) * -1
            
        else:
            ypred_val = np.ones((X.shape[0],1))
            y_alpha = self.y_alpha.reshape(1,-1)

            kernel = form_kernel(self.Xt,X,self.gamma)
            wtphi = np.matmul(y_alpha,kernel)
            wtphi += self.b
            ypred_val = (wtphi > 0) * 1 + (wtphi < 0) * -1
        return ypred_val

    def predict_score(self,X):

        "This function returns the value of |W.X + B|."
        
        if self.k == "Linear":
            wtphi = (np.matmul(X,self.w.T) + self.b)
            ypred_val = (wtphi > 0) * 1 + (wtphi < 0) * -1

        else:
            ypred_val = np.ones((X.shape[0],1))
            y_alpha = self.y_alpha.reshape(1,-1)

            kernel = form_kernel(self.Xt,X,self.gamma)
            wtphi = np.matmul(y_alpha,kernel)
            wtphi += self.b
            
        return wtphi 

if __name__ == "__main__":

    train_data = sys.argv[1]
    test_data = sys.argv[2]
    out_file = sys.argv[3]

    #data_dir = "../data/fashion_mnist/"
    start = time.time()

    train_df = pd.read_csv(train_data, header = None)
    #val_df = pd.read_csv(train_data_dir + "val.csv")
    test_df = pd.read_csv(test_data, header = None)


    X = train_df[train_df.columns[:-1]].values
    y = train_df[train_df.columns[-1]].values

    X = X / 255

    # '''
    # Using Liblinear for Hyperparameter tuning, and using the optimal hyperparameter value for training of my custom implementation.
    # '''

    Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size = 0.25)

    Cs = [0.1, 1.0, 5.0, 10, 20, 50, 100]
    scores = {}
    for c in Cs:
        #print(c)
        svm = SVC(C = c, gamma=0.05)
        start = time.time()
        svm.fit(Xtrain,ytrain)
        end = time.time()
        scores[c] = svm.score(Xval,yval)
        #print("C:",c,scores[c])
    
    '''
    The key-value pair corresponding to the maximum accuracy is selected, and used for training over the entire data.
    '''
    max_acc = 0
    max_key = None
    for k in scores.keys():
        if scores[k] > max_acc:
            max_key = k
            max_acc = scores[k]

    classifiers = {}

    "Training all the 10C2 classifiers."
    
    for i in range(10):
        for j in range(i+1,10):
            Xtrain,ytrain= process_data(train_df,i,j)
            model = SVM(kernel = "Gaussian",gamma = 0.05, c = 1)
            model.fit(Xtrain,ytrain)
            classifiers[(i,j)] = model
            end = time.time()
            #print(i,j,end-start)

    #end = time.time()
    # print("Training Complete, Time:",end-start)

    #print(test_df.shape)
    Xtest = test_df[test_df.columns[:-1]].values
    ytest = test_df[test_df.columns[-1]].values
    Xtest = Xtest / 255

    #print(ytest.shape)
    #print(Xtest.shape)

    #print("Accuracy:",model.score(Xtest,ytest))

    #Xval = val_df[val_df.columns[:-1]].values
    #yval = val_df[val_df.columns[-1]].values
    #Xval = Xval / 255

    # start = time.time()
    # acc = compute_score(Xval,yval)
    # end = time.time()
    # print("Validation Inference Time:",end-start)
    # print("Validation Accuracy:",acc)

    # start = time.time()
    # acc = compute_score(Xtest,ytest)
    end = time.time()
    #print("Test Inference Time:",end-start)
    #print("Test Accuracy:",model.score(Xtest,ytest))

    acc = compute_score(Xtest,ytest,classifiers,classes = True)
    #print(acc.shape)
    write_predictions(out_file,acc)

import numpy as np
import pandas as pd
import time
import sys

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_grad(z):
    e = sigmoid(z)
    return e * (1 - e)

def relu(z):
    z[z <= 0] = 0
    return z

def relu_grad(z):
    return (z > 0) * 1

def binarize(y,num_classes):
    temp = np.zeros((y.shape[0],num_classes))
    for j in range(len(y)):
        for i in range(num_classes):
            if i == y[j]:
                temp[j][i] = 1
                break
    return temp

class Layer:
    def __init__(self,in_size,out):
        
        self.input = None
        self.output = None
        self.wt = np.random.normal(0,0.01,(out,in_size))
        self.grads = None
        self.inputs = None
        self.outputs = None
        self.grad = None
        
class NeuralNetwork:
    
    def __init__(self, M, n, r, layers = None, activation = 'sigmoid', verbose = False, adaptive = False):
        self.target = r
        self.layers = []
        self.mini_batch_size = M
        
        self.verbose = verbose
        self.features = n+1
        self.adaptive = adaptive

        if activation == 'sigmoid':
            self.activation = sigmoid
            self.gradient = sigmoid_grad
            self.lr = 0.001

        elif activation == 'ReLU':
            self.activation = relu
            self.gradient = relu_grad
            self.lr = 0.0001
        
        down = n+1
        up = layers[0]
        
        self.layers.append(Layer(down,up))
        
        for i in range(len(layers)-1):
            down = layers[i]
            up = layers[i+1]
            self.layers.append(Layer(down,up))
            
        self.layers.append(Layer(layers[-1],r))
        
    def forward(self,Xt):
        
        X = np.ones((Xt.shape[0],Xt.shape[1]+1))
        X[:,1:] = X[:,1:] * Xt
        
        self.layers[0].input = X
        
        for i in range(1,len(self.layers)):
            
            prev_layer = self.layers[i-1]
            curr_layer = self.layers[i]
            z = np.matmul(prev_layer.input,prev_layer.wt.T)
            
            if i == len(self.layers):
                act = sigmoid(z)
            else:
                act = self.activation(z)
            curr_layer.input = act
        
        last_layer = self.layers[-1]
        last_layer.output = sigmoid(np.matmul(last_layer.input,last_layer.wt.T))

    def backward(self,ylabel):

        ypred = self.layers[-1].output
        error = np.mean(0.5*np.sum((ypred-ylabel)**2,axis = 1))
        
        del_netj = -(ylabel-ypred) * sigmoid_grad(ypred)
        
        for i in range(len(self.layers)-1,-1,-1):
            
            curr_layer = self.layers[i]
            curr_layer.grad = np.matmul(del_netj.T,curr_layer.input)
            del_netj = np.matmul(del_netj, curr_layer.wt) * self.gradient(curr_layer.input)            
    
        return error
    
    def update(self,epoch):
        for layer in self.layers:
            if self.adaptive:
                layer.wt = layer.wt - (self.lr)/np.sqrt(epoch+1) * layer.grad
            else:
                layer.wt = layer.wt - self.lr * layer.grad
    
    def train(self,X,y):
        acc_train = []
        acc_test = []
        b = self.mini_batch_size
        m = X.shape[0]
        batchesX = []
        batchesy = []
        
        for i in range(m//b):
            batchesX.append(X[b*i:b*i+b])
            batchesy.append(y[b*i:b*i+b])
            
        converged = False
        temp = []
        epochs = 0
        while not converged:
            errs = []
            for batch in range(m//b):
                Xbatch,ybatch = batchesX[batch],batchesy[batch]
                self.forward(Xbatch)
                err = self.backward(ybatch)
                self.update(epochs)
                errs.append(err)
            
            temp.append(np.mean(errs))
            if len(temp) > 5:
                if np.max(temp[-5:]) - np.min(temp[-5:]) < 1e-4:
                    converged = True
                    return temp[-1]
            
            if self.verbose:
                print("Epochs:",epochs,"Loss:",np.mean(errs))
            epochs += 1
        
    def predict(self,X):
        self.forward(X)
        return np.argmax(self.layers[-1].output,axis = 1)

    def score(self,X,y):
        return np.mean(y == self.predict(X))


if __name__ == '__main__':

    Xtrain = np.load(sys.argv[1])
    ytrain = np.load(sys.argv[2])
    Xtest = np.load(sys.argv[3])
    ytest = np.load('../data/neural_network_kannada/y_test.npy')

    Xtrain = Xtrain.reshape((Xtrain.shape[0],-1))
    Xtest = Xtest.reshape((Xtest.shape[0],-1))

    output_file = sys.argv[4]
    batch_size = int(sys.argv[5])
    hidden_layer_list = [int(i) for i in sys.argv[6].split()]
    activation = sys.argv[7]

    ybin = binarize(ytrain,10)

    if activation == 'relu':
        nn = NeuralNetwork(batch_size, Xtrain.shape[1], 10, activation = 'ReLU', layers = hidden_layer_list, verbose = False, adaptive=False)
    else:
        nn = NeuralNetwork(batch_size, Xtrain.shape[1], 10, activation = 'sigmoid', layers = hidden_layer_list, verbose = False, adaptive=False)
    
    start = time.time()
    nn.train(Xtrain,ybin)
    end = time.time()
    #print("Time:",end-start)

    ypred = nn.predict(Xtest)
    
    print("Score-Test:",nn.score(Xtest,ytest))

    write_predictions(output_file,ypred)

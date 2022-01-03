import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from q2 import NeuralNetwork, binarize

Xtrain = np.load('../data/neural_network_kannada/X_train.npy')
ytrain = np.load('../data/neural_network_kannada/y_train.npy')
Xtest = np.load('../data/neural_network_kannada/X_test.npy')
ytest = np.load('../data/neural_network_kannada/y_test.npy')

Xtrain = Xtrain.reshape((Xtrain.shape[0],-1))
Xtest = Xtest.reshape((Xtest.shape[0],-1))

metric_layer = []
#times = []
layers = [1, 10, 50, 100, 500]
accs_train = []
accs_test = []
ybin = binarize(ytrain,10)

nn = NeuralNetwork(100, Xtrain.shape[1],10,layers = [1],verbose = False)
start = time.time()
acc_train,acc_test = nn.train(Xtrain,ybin)
accs_train.append(acc_train)
accs_test.append(acc_test)
end = time.time()
print("Time:",end-start)

'''Code for graph-plotting'''

fig,ax = plt.subplots()
ax.plot(layers[1:],scores_test1[1:],label = "Test Set - Adaptive Learning Rate")
ax.plot(layers[1:],scores_train1[1:],label = "Train Set - Adaptive Learning Rate")
ax.plot(layers[1:],scores_test[1:],label = "Test Set - Constant Learning Rate")
ax.plot(layers[1:],scores_train[1:],label = "Train Set - Constant Learning Rate")
ax.set_xlabel("No.of neurons in the hidden layer")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs # Neurons")
ax.legend(loc = 'best')
fig.savefig("../output/q2/ad_acc_final.png",dpi = 200)

fig,ax = plt.subplots()
ax.plot(layers,times1,label = 'Adaptive Learning Rate')
ax.plot(layers,times,label = 'Constant Learning Rate')
ax.set_xlabel("No.of neurons in the hidden layer")
ax.set_ylabel("Training Time (in s)")
ax.set_title("Training Time vs # Neurons")
ax.legend(loc = 'best')
fig.savefig("../output/q2/ada_time.png",dpi = 200)

fig,ax = plt.subplots()
ax.plot(layers,metric_layer1,label = 'Adaptive Learning Rate')
ax.plot(layers,metric_layer,label = 'Constant Learning Rate')
ax.set_xlabel("No.of neurons in the hidden layer")
ax.set_ylabel("MSE at convergence")
ax.set_title("Plot of Metric vs # Neurons")
ax.legend(loc = 'best')
fig.savefig("../output/q2/ad_metric.png",dpi = 200)


from sklearn.neural_network import MLPClassifier

start = time.time()
mlp = MLPClassifier((100,100),activation='relu',solver='sgd',batch_size=100,verbose = True,learning_rate='adaptive')
mlp.fit(Xtrain,ytrain)
end = time.time()
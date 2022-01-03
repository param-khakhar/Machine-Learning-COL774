import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from q1 import DecisionTree

train_df = pd.read_csv('../data/decision_tree/train.csv')
val_df = pd.read_csv('../data/decision_tree/val.csv')
test_df = pd.read_csv('../data/decision_tree/test.csv')

Xtrain = train_df[train_df.columns[:-1]]
ytrain = train_df[train_df.columns[-1]]
Xval = val_df[val_df.columns[:-1]]
yval = val_df[val_df.columns[-1]]
Xtest = test_df[test_df.columns[:-1]]
ytest = test_df[test_df.columns[-1]]

# Code for training dtr using the iterative function
start = time.time()
dtr_iter = DecisionTree()
dtr_iter.iter_fit(Xtrain,ytrain,Xtest,ytest,Xval,yval)
end = time.time()
print("Training Time:",end-start)

'''
Code for plotting accuracy on validation and train set while building the tree.
'''

fig,ax = plt.subplots()
ax.plot([i for i in range(len(dtr_iter.acc_train))],dtr_iter.acc_train,label = "Train Accuracy")
ax.plot([i for i in range(len(dtr_iter.acc_test))],dtr_iter.acc_test,label = "Test Accuracy")
ax.plot([i for i in range(len(dtr_iter.acc_val))],dtr_iter.acc_val,label = "Validation Accuracy")
ax.legend(loc = 'best')
ax.set_xlabel("No. of nodes in 1000")
ax.set_ylabel("Accuracy")
ax.set_title("Decision Tree Accuracy on adding nodes")
fig.savefig('plot_dtr.png',dpi = 200)
plt.show()

'''
Code for plotting graph for accuracies while pruning.
'''
fig,ax = plt.subplots()
ax.plot([i for i in range(len(train_prune))],train_prune,label = "Train Accuracy")
ax.plot([i for i in range(len(val_prune))],val_prune,label = "Validation Accuracy")
ax.plot([i for i in range(len(test_prune))],test_prune,label = "Test Accuracy")
ax.legend(loc = 'best')
ax.set_xlabel("No. of nodes in 200's")
ax.set_ylabel("Accuracy")
ax.set_title("Decision Tree Accuracy on pruning nodes")
fig.savefig('../output/q1/plot_dtr_prune.png',dpi = 200)
plt.show()

'''Code for 1c'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, GridSearchCV, PredefinedSplit

param_grid = {'n_estimators': [i for i in range(50,500,100)],'max_features':[i for i in np.arange(0.1,1,0.2)],'min_samples_split':[i for i in range(2,12,2)]}
def oob_scorer(estimator, X, y):
    return estimator.oob_score_

rfc = RandomForestClassifier(oob_score = True, criterion = 'entropy')

gs1 = GridSearchCV(estimator=rfc,param_grid=param_grid,scoring=oob_scorer,cv = PredefinedSplit([-1]*(Xtrain.shape[0]-1) + [0]),n_jobs=1, verbose = 10)
gs1.fit(Xtrain, ytrain)

'''Code for 1d'''

n_estimators = 450
min_samples_split = 2
max_features = 0.7

val_acc = []
test_acc = []
estimator = []
start = time.time()

for i in range(50,500,100):
    rf = RandomForestClassifier(n_estimators = i, min_samples_split=2, max_features=0.7,criterion = 'entropy')
    rf.fit(Xtrain,ytrain)
    val_acc.append(rf.score(Xval,yval))
    test_acc.append(rf.score(Xtest,ytest))
    estimator.append(i)
    print(i)
    end = time.time()
    print("Time:",end-start)

fig,ax = plt.subplots(figsize = (8,4))
ax.plot(estimator,val_acc,label = 'Validation Accuracy')
ax.plot(estimator,test_acc,label = 'Test Accuracy')
ax.set_xlabel("No. of Estimators (Optimal : 450)")
ax.set_title("Varying the number of Estimators, max_features = 0.7, min_samples_split = 2")
ax.legend(loc = 'best')
fig.savefig('../output/q1/estimator.png',dpi = 200)


n_estimators = 450
min_samples_split = 2
max_features = 0.7

val_acc_1 = []
test_acc_1 = []
estimator_1 = []
start = time.time()
for i in range(2,12,2):
    rf = RandomForestClassifier(n_estimators = 450, min_samples_split=i, max_features=0.7,criterion = 'entropy')
    rf.fit(Xtrain,ytrain)
    val_acc_1.append(rf.score(Xval,yval))
    test_acc_1.append(rf.score(Xtest,ytest))
    estimator_1.append(i)
    end = time.time()
    print(i)
    print("Time:",end-start)


fig,ax = plt.subplots(figsize = (8,4))
ax.plot(estimator_1,val_acc_1,label = 'Validation Accuracy')
ax.plot(estimator_1,test_acc_1,label = 'Test Accuracy')
ax.set_xlabel("Min_samples_split (Optimal : 2)")
ax.set_title("Varying the number of min_samples_split, max_features = 0.7, estimators = 450")
ax.legend(loc = 'best')
fig.savefig('../output/q1/min_samples_split.png',dpi = 200)


n_estimators = 450
min_samples_split = 2
max_features = [0.1,0.3,0.5,0.7,0.9]

val_acc_2 = []
test_acc_2 = []
estimator_2 = []
start = time.time()
for i in max_features:
    rf = RandomForestClassifier(n_estimators = 450, min_samples_split=2, max_features=i,criterion = 'entropy')
    rf.fit(Xtrain,ytrain)
    val_acc_2.append(rf.score(Xval,yval))
    test_acc_2.append(rf.score(Xtest,ytest))
    estimator_2.append(i)
    end = time.time()
    print(i)
    print("Time:",end-start)

fig,ax = plt.subplots(figsize = (8,4))
ax.plot(estimator_2,val_acc_2,label = 'Validation Accuracy')
ax.plot(estimator_2,test_acc_2,label = 'Test Accuracy')
ax.set_xlabel("Max-Features (Optimal : 0.7)")
ax.set_title("Varying max-features, min_samples_split = 2, estimators = 450")
ax.legend(loc = 'best')
fig.savefig('../output/q1/max_features.png',dpi = 200)
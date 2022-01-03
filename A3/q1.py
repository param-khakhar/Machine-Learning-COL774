import numpy as np
import pandas as pd
import time
import sys

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

def entropy(y):
    # Computes H(y) for the dataset provided
    unique_val = np.unique(y)
    count = []
    for v in unique_val:
        count.append(np.count_nonzero(y == v))
    
    count = np.array(count)
    count = count / np.sum(count)
    
    return np.sum(-count * np.log2(count))

class Node:
    
    def __init__(self,attribute = None, label = None):
        self.attribute = attribute              # Attribute (index) on which the node is splitted.
        self.children = {}                      # List of children where the individual elements are Nodes.
        self.label = label                      # The majority class at that attribute.
        self.left = None                        # Left child in case of splitting a continuous attribute. Contains values of Xj lesser than the median.
        self.right = None                       # Right child in case of splitting a continuous attriute. Contains values of Xj greater than the median.
        self.cont = False                       # True if the attribute is continuous
        self.value = None                       # Value of the attribute which is used for making the split.
        self.parent = None                      # The parent pointer for the nodes.
        self.isLeaf = False                     # Boolean variable indicating whether the node is leaf or not.
        self.id = None                          # Unique id assigned to each node
        self.inf_gain = None                    # The information_gain at this node

    
    def __lt__(self, other):
        return self.inf_gain > other.inf_gain

def CreateLeaf(label):
    n = Node(label = label)
    return n

class DecisionTree:
    
    def __init__(self):
        
        self.root = None
        self.cont = None
        self.disc = None
        self.acc_test = []
        self.acc_val = []
        self.acc_train = []
    
    def ChooseBestAttrToSplit(self,X,y):
        
        cols = X.shape[1]
        inf_gain = -np.inf
        bestattr = -1
        h_y = entropy(y)
        for i in range(cols):
            h_y_x = 0
            if np.count_nonzero(self.cont == i):
                med = np.median(X[:,i])
                
                mask1 = (X[:,i] <= med)
                h_y_x += np.mean(mask1)*entropy(y[mask1])
    
                mask2 = (X[:,i] > med)
                h_y_x += np.mean(mask2)*entropy(y[mask2])
            
            else:
                # Discrete Feature
                values = np.unique(X[:,i])
                for v in values:
                    mask = (X[:,i] == v)
                    h_y_x +=  np.mean(mask)*entropy(y[mask])
                
            diff = h_y - h_y_x
            if diff > inf_gain:
                inf_gain = diff
                bestattr = i
        return bestattr,inf_gain
            
    def GrowTree(self,X,y):
        
        if np.unique(y).shape[0] == 1:
            #print("Here")
            return CreateLeaf(label = y[0])

        Xj,inf_gain = self.ChooseBestAttrToSplit(X,y)
        #print(Xj,height)
        sepCol = X[:,Xj]
        parent = Node(attribute=Xj)
        parent.inf_gain = inf_gain
        
        if np.count_nonzero(self.cont == Xj):
            
            # Splitting for Continuous features
            
            sep = np.median(sepCol)
            mask1 = (X[:,Xj] <= sep)
            mask2 = (X[:,Xj] > sep)
            if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                
                subX = X[mask1]
                subY = y[mask1]
                child = self.GrowTree(subX,subY)
                parent.left = child
                child.parent = parent
                
                subX = X[mask2]
                subY = y[mask2]

                child = self.GrowTree(subX,subY)
                parent.right = child
                child.parent = parent

                parent.value = sep
                parent.cont = True
                
            else:
                # Termination condition where there are non-empty left and right splits.
                val = np.bincount(y)
                parent.label = np.argmax(val)
            
        else:
            
            # Splitting for discrete features.
            
            unique = np.unique(sepCol)
            for i in unique:
                mask = (X[:,Xj] == i)
                subX = X[mask]
                subY = y[mask]
                child = self.GrowTree(subX,subY)
                child.parent = parent
                parent.children[i] = child
                
        return parent
    
    def iter_GrowTree(self,X,y,Xtest,ytest,Xval,yval):
        
        # Iterative implementation of the recursive code.

        Xstack = []
        ystack = []
        childStack = []
        root = Node()
        childStack.append(root)
        Xstack.append(X)
        ystack.append(y)
        nodes = 0
        
        while len(childStack) > 0:

            curr = childStack.pop()
            
            currX = Xstack.pop()
            curry = ystack.pop()
            
            if np.unique(curry).shape[0] == 1:
                curr.label = curry[0]
                curr.isLeaf = True

            else:
                Xj,inf_gain = self.ChooseBestAttrToSplit(currX,curry)
                curr.inf_gain = inf_gain
                curr.attribute = Xj
                curr.label = np.argmax(np.bincount(curry))
                sepCol = currX[:,Xj]

                if np.count_nonzero(self.cont == Xj):

                    sep = np.median(sepCol)
                    mask1 = (currX[:,Xj] <= sep)
                    mask2 = (currX[:,Xj] > sep)
                    if np.sum(mask1) > 0 and np.sum(mask2) > 0:

                        subX = currX[mask2]
                        subY = curry[mask2]

                        child = Node()
                        curr.right = child
                        child.parent = curr

                        childStack.append(child)
                        Xstack.append(subX)
                        ystack.append(subY)

                        subX = currX[mask1]
                        subY = curry[mask1]
                        child = Node()
                        curr.left = child
                        child.parent = curr

                        childStack.append(child)
                        Xstack.append(subX)
                        ystack.append(subY)

                        curr.value = sep
                        curr.cont = True
                    else:
                        val = np.bincount(y)
                        curr.label = np.argmax(val)
                else:
                    unique = np.unique(sepCol)
                    for i in unique:
                        mask = (currX[:,Xj] == i)
                        subX = currX[mask]
                        subY = curry[mask]
                        child = Node()
                        curr.children[i] = child
                        child.parent = curr
                        childStack.append(child)
                        Xstack.append(subX)
                        ystack.append(subY)
                
            #print(nodes)
            #if nodes % 1000 == 0:
                #ypred = predictions(root,Xtrain)
                #acc = np.mean(ypred == y)
                #print("Train:",acc)
                #self.acc_train.append(acc)
                #ypred = predictions(root,Xtest)
                #acc = np.mean(ypred == ytest.values)
                #print("Test:",acc)
                #self.acc_test.append(acc)
                #ypred = predictions(root,Xval)
                #acc = np.mean(ypred == yval.values)
                #print("Val:",acc)
                #self.acc_val.append(acc)
                #print(len(childStack),len(ystack))
            curr.id = nodes
            nodes += 1
            
        return root
            
            
    def fit(self,X,y):
        # X and y are pandas dataframes with the column labels stating whether a feature is continuous or discrete
        
        #Segreagating continuous and discrete features and storing their indices
        
        cols = X.columns
        cont = []
        for i in range(len(cols)):
            if cols[i][-8:] == "Discrete":
                continue
            else:
                cont.append(i)
        self.cont = np.array(cont)
        self.root = self.GrowTree(X.values,y.values,0)
        
    def iter_fit(self,X,y,Xtest,ytest,Xval,yval):
        # Function for fitting to the iterative version of train. Also used for generating the data for the plot.
        cols = X.columns
        cont = []
        for i in range(len(cols)):
            if cols[i][-8:] == "Discrete":
                continue
            else:
                cont.append(i)
        self.cont = np.array(cont)
        self.root = self.iter_GrowTree(X.values,y.values,Xtest,ytest,Xval,yval)
        
        
    def predict(self,X):
        # Function for predicting the labels for the instances. Returns an np array corresponding of size X.shape[0]
        X = X.values
        ypred = []
        m = X.shape[0]
        for i in range(m):
            x = X[i]
            curr = self.root
            while curr and (curr.left or curr.right or len(curr.children) > 0):

                Xj = curr.attribute
                #print(Xj)
                if curr.cont:
                    if x[Xj] <= curr.value:
                        curr = curr.left
                    else:
                        curr = curr.right
                else:
                    curr = curr.children[x[Xj]]
            
            if curr:
                ypred.append(curr.label)
            else:
                ypred.append(-1)
                
        return np.array(ypred)
    
    def score(self,X,y):
        # Returns the accuracy score for the predictions made
        ypreds = self.predict(X)
        return np.mean(ypreds == y.values)


'''
    Functions used for pruning decision trees.
'''

def createNodeList(root,nodeError):
    # This function fills the dictionary whose keys are the ids of the nodes and the values are intialized to 0.
    nodeError[root.id] = 0
    if root.cont:
        createNodeList(root.left,nodeError)
        createNodeList(root.right,nodeError)
    else:
        for child in root.children.keys():
            createNodeList(root.children[child],nodeError)

def countLeaves(root):
    # Function for counting the number of leaves in the tree
    if root.isLeaf:
        return 1
    else:
        n = 0
        if root.cont:
            n += countLeaves(root.left)
            n += countLeaves(root.right)
        else:
            for child in root.children.keys():
                n += countLeaves(root.children[child])
        return n

def isTwig(root):
    # A Twig is a node whose all children are leaves.
    if not root.isLeaf:
        if root.cont:
            if (root.left.isLeaf) and (root.right.isLeaf):
                return True
            return False
        else:
            for child in root.children.keys():
                if not root.children[child].isLeaf:
                    return False
            return True
    else:
        return False

def classifyValidationDataInstance(root,x,y,nodeError):
    # Each validation instance is passed down the tree and error is checked for each node if the predicted class is the majority label assigned to each node.
    if root.label != y:
        nodeError[root.id] += 1
    if not root.isLeaf:
        if root.cont:
            if x[root.attribute] <= root.value:
                child = root.left
            else:
                child = root.right
        else:
            child = root.children[x[root.attribute]]
        classifyValidationDataInstance(child,x,y,nodeError)
    return 

def classifyValidationData(root,Xval,yval):
    # Errors computed for all the instances in the validation data.
    Xval = Xval.values
    yval = yval.values
    nodeError = {}
    createNodeList(root,nodeError)
    m = Xval.shape[0]
    for i in range(m):
        x = Xval[i]
        y = yval[i]
        classifyValidationDataInstance(root,x,y,nodeError)
    return nodeError

def collectTwigsByErrorCount(root,nodeError,heap):
    # For all the twigs, we calculate the error if we prune the children of the twigs. We insert the twig into a min-heap and the key for the
    # heap is the difference between the errors of the parent and child.
    if isTwig(root):
        twigErrorIncrease = nodeError[root.id]
        
        if root.cont:
            twigErrorIncrease -= nodeError[root.left.id]
            twigErrorIncrease -= nodeError[root.right.id]
            #heappush(heap,(twigErrorIncrease,root))
            heap.append((twigErrorIncrease,root))
        else:
            for child in root.children.keys():
                twigErrorIncrease -= nodeError[root.children[child].id]
            heap.append((twigErrorIncrease,root))
            #heappush(heap, (twigErrorIncrease,root))
    else:
        if root.cont:
            collectTwigsByErrorCount(root.left,nodeError,heap)
            collectTwigsByErrorCount(root.right,nodeError,heap)
        else:
            for child in root.children.keys():
                collectTwigsByErrorCount(root.children[child],nodeError,heap)
    return heap

def pruneByClassification(root,Xval,yval):
    # We prune out nodes until the key of the nodes(twigs) in the heap don't become >= 0. 
    nodeError = classifyValidationData(root,Xval,yval)
    twigHeap = collectTwigsByErrorCount(root,nodeError,[])
    totalLeaves = countLeaves(root)
    
    c = 0
    
    while True:
        
        twig = min(twigHeap)
        twigHeap.remove(twig)
        #print(c,twig[0])
        if twig[0] > 0:
            break
        par = twig[1].parent
        if twig[1].cont:
            totalLeaves -= 1
            
            twig[1].left = None
            twig[1].right = None
            twig[1].isLeaf = True
            twig[1].inf_gain = 0
            
        else:
            totalLeaves -= len(twig[1].children)-1
            twig[1].children = {}
            twig[1].isLeaf = True
            twig[1].inf_gain = 0
            
        if isTwig(par):
            twigErrorIncrease = nodeError[par.id]
            if par.cont:
                twigErrorIncrease -= nodeError[par.left.id]
                twigErrorIncrease -= nodeError[par.right.id]
            else:
                for child in par.children.keys():
                    twigErrorIncrease -= nodeError[par.children[child].id]

            twigHeap.append((twigErrorIncrease,par))        
            #heappush(twigHeap,(twigErrorIncrease,par))
        # if c % 200 == 0:
        #     val_prune.append(dtr_iter.score(Xval,yval))
        #     train_prune.append(dtr_iter.score(Xtrain,ytrain))
        #     test_prune.append(dtr_iter.score(Xtest,ytest))
        c += 1
    #print(c)
    return

if __name__ == "__main__":

    q = sys.argv[1]
    train_df = pd.read_csv(sys.argv[2])
    val_df = pd.read_csv(sys.argv[3])
    test_df = pd.read_csv(sys.argv[4])
    output_file = sys.argv[5]

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
    #print("Training Time:",end-start)

    if q == '2':
        #print("Here")
        pruneByClassification(dtr_iter.root,Xval,yval)

    ypred = dtr_iter.predict(Xtest)
    #print("Score-Test:",dtr_iter.score(Xtest,ytest))
    #print("Score-Val:",dtr_iter.score(Xval,yval))
    #print("Score-Train:",dtr_iter.score(Xtrain,ytrain))

    write_predictions(output_file,ypred)







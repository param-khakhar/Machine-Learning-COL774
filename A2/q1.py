import sys
import pandas as pd
import numpy as np
import time
#from utils import getStemmedDocuments, json_reader, tokenize
import json
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt

def json_reader(fname):
    """
        Read multiple json files
        Args:
            fname: str: input file
        Returns:
            generator: iterator over documents 
    """
    for line in open(fname, mode="r"):
        yield json.loads(line)

def tokenize(doc, return_tokens = True):

    #tokens = word_tokenize(doc.lower())

    '''
        Removing all the punctuation marks!
    '''
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(doc.lower())

    en_stop = set(stopwords.words('english'))

    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    #stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)

    if not return_tokens:
        return ' '.join(stopped_tokens)
    return list(stopped_tokens)

def _stem(doc, p_stemmer, en_stop, return_tokens):

    '''
        Removing all the punctuation marks!
    '''
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(doc.lower())

    #stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer.stem(token), tokens)
    if not return_tokens:
        return ' '.join(stemmed_tokens)
    return list(stemmed_tokens)

def getStemmedDocuments(docs, return_tokens=True):
    """
        Args:
            docs: str/list(str): document or list of documents that need to be processed
            return_tokens: bool: return a re-joined string or tokens
        Returns:
            str/list(str): processed document or list of processed documents
        Example: 
            new_text = "It is important to by very pythonly while you are pythoning with python.
                All pythoners have pythoned poorly at least once."
            print(getStemmedDocuments(new_text))
        Reference: https://pythonprogramming.net/stemming-nltk-tutorial/
    """
    en_stop = set(stopwords.words('english'))
    ps = PorterStemmer()
    #sno = SnowballStemmer('english')
    #p_stemmer = lru_cache(maxsize=None)(ps.stem)
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(_stem(item, ps, en_stop, return_tokens))
        return output_docs
    else:
        return _stem(docs, ps, en_stop, return_tokens)

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

if __name__ == "__main__":

    train_data = sys.argv[1]
    test_data = sys.argv[2]
    output_file = sys.argv[3]

    train = [i for i in json_reader(train_data)]
    test = [i for i in json_reader(test_data)]
    

    start = time.time()
    '''
    The code for preprocessing the input data. Each review, is first tokenized 
    into words, along with the removal of punctuations. Out of these tokens,
    stopwords are filtered out, and each word is stemmed to it's root form.
    There's a dictionary corresponding to each document which stores the word(key)
    and the number of occurrences (value). 
    
    stemmed_docs is the list of all the dictionaries for the documents.
    The dictionary inv_dict, represents the vocabulary and represents every unique
    word by an index.
    '''
    stemmed_docs = []
    inv_dict = {}

    k = 0
    j = 0
    
    for doc in train:
        stemmed_doc = getStemmedDocuments(doc['text'])
        temp = {}
        for word in stemmed_doc:
            if word not in inv_dict:
                inv_dict[word] = k
                k += 1
            if word not in temp:
                temp[word] = 1
            else:
                temp[word] += 1
        stemmed_docs.append((len(stemmed_doc),temp))

    #print(len(inv_dict))
    end = time.time()
    #print("Time Preprocessing:",end-start)

    labels = [0 for i in range(5)]
    for i in train:
        stars = i['stars']
        labels[int(stars)-1] += 1
    #print(labels)

    '''
    Learning Parameters:
    phi_params : They would be simply the fraction of the respective classes among the entire training data set.
    theta_params : Theta_Params would also be calculated for every unique word in the dictionary and also for all the available classes.
    '''
    theta_params = []
    phi_params = []

    total = len(train)
    for i in range(len(labels)):
        phi_params.append(labels[i]/total)
    #print(phi_params)

    #start = time.time()
    v = len(inv_dict)
    k = len(labels)
    m = len(train)

    resp_labels = [0 for i in range(k)]
    theta_params = [[1 for i in range(v)] for j in range(k)]

    for i in range(m):
        length,tokens = stemmed_docs[i]
        label = int(train[i]['stars'])
        
        for token in tokens.keys():
            if tokens[token] < 10:
                theta_params[label-1][inv_dict[token]] += tokens[token]
            
        resp_labels[label-1] += length

    theta_params = np.array(theta_params)

    #print(theta_params.shape)
    end = time.time()
    #print("Time Learning:",end-start)
    
    '''
    Inference: During, inference the reviews in the test dataset undergo the same
    preprocessing as the training dataset, and for all the tokens, in the review,
    log of the probability is computed for each of the class.

    The list classes stores the log of the probability for each document, and the index
    with the maximum value of the value is assigned as the label.

    For the words not in the vocabulary, the KeyError is catched in the try except block
    and the uniform probability consistent with the smoothing is done.
    '''

    true_labels = [int(i['stars']) for i in test]
    #print(len(true_labels))

    #start = time.time()
    pred_labels = []
    proba = []
    for docs in test:
        doc = docs['text']
        classes = [1.0 for i in range(k)]
        stemmed_doc = getStemmedDocuments(doc)
        
        for token in stemmed_doc:
            for j in range(k):
                try:
                    classes[j] += np.log(theta_params[j][inv_dict[token]]) - np.log(v+resp_labels[j])
                except KeyError:
                    classes[j] -= np.log(resp_labels[j]+v)

        for j in range(k):
            classes[j] += np.log(phi_params[j])

        temp = []
        for j in range(k):
            temp.append(classes[j])
        
        temp = np.array(temp)
        temp -= np.max(temp)
        temp = np.exp(temp)
        temp /= np.sum(temp)
        proba.append(list(temp))
        pred_labels.append(np.argmax(classes)+1)

    end = time.time()
    #print("Time Inference:",end-start)

    true = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == true_labels[i]:
            true+=1

    #print("Accuracy:",true/len(pred_labels))
    write_predictions(output_file,pred_labels)


    '''
    # Code for random prediction of labels.

    import random

    stars = [1,2,3,4,5]

    random_pred = [random.choice(stars) for i in range(len(test))]
    true = 0
    for i in range(len(test)):
        if random_pred[i] == true_labels[i]:
            true+=1
    print(true)
    print(true/len(test))
    '''



    '''
    # The majority of the reviews had a rating of 5 stars, therefore, the code for predicting the accuracy, on assigning 5 star to each review.

    true = 0
    for i in range(len(test)):
        if 5 == true_labels[i]:
            true+=1
    print(true)
    print(true/len(test))
    '''


    '''
    # Code for plotting ROC_AUC curves

    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from matplotlib import pyplot as plt
    from scipy import interp

    true = np.array(true_labels)
    true -= 1
    y_test = label_binarize(true, [i for i in range(5)])

    Code for plotting ROC curves, referred from https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

    new = np.array(proba)   # Proba is the normalized probability for each class for each instance of the test set.

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(k):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], new[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), new.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(k)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(k):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= k

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr

    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    for i in range(k):
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i] + ' label='+str(i+1))
        plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr['micro'],tpr['micro'],label = 'ROC curve (area = %0.2f)' % roc_auc['micro'] + ' Micro-Average')
    plt.plot(fpr['macro'],tpr['macro'],label = 'ROC curve (area = %0.2f)' % roc_auc['macro'] + ' Macro-Average')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig("plot.png",dpi = 200)
    '''

    

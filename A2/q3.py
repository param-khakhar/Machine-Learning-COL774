import sys
import pandas as pd
import numpy as np
import time
import nltk
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

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

    #data_dir = "../data/col774_yelp_data/"
    #output_dir = "../output/q1/"

    train = [i for i in json_reader(train_data)]
    test = [i for i in json_reader(test_data)]

    #print(len(train),len(test))

    '''
    Store the input reviews and the corresponding ratings in a list.
    '''

    start = time.time()
    X = [i['text'] for i in train]
    Xtest = [i['text'] for i in test]
    y = [int(i['stars']) for i in train]
    ytest = [int(i['stars']) for i in test]
    end = time.time()
    #print("Time:",end-start)


    '''
    Using Stemming along with StopWords Removal, and converting the resulting tokens into TF-IDF vectors.
    '''

    stemmer = PorterStemmer()
    analyzer = CountVectorizer().build_analyzer()

    def stemmed_words(doc):
        return (stemmer.stem(w) for w in analyzer(doc))

    start = time.time()
    stem_vectorizer = CountVectorizer(analyzer=stemmed_words,stop_words='english')
    Xcounts = stem_vectorizer.fit_transform(X)
    Xtcounts = stem_vectorizer.transform(Xtest)
    tfidf_vect = TfidfTransformer()
    X = tfidf_vect.fit_transform(Xcounts)
    Xtest = tfidf_vect.transform(Xtcounts)
    end = time.time()
    #print("Preprocessing Time:",end-start)

    # Train-Validation Ratio : 75% - 25%

    Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size = 0.25)

    '''
    No Hyper-parameters to be tuned for Multinomial Naive Bayes
    '''

    # start = time.time()
    # mnb = MultinomialNB()
    # mnb.fit(X,y)
    # end = time.time()
    # #print("Training Time for Multinomial Naive Bayes:",end-start)
    # #print("NB:",mnb.score(Xval,yval))

    '''
    Choosing the suitable value of C among {1e-3, 1e-2, 1e-1, 1, 10, 100} for Liblinear SVM
    Validation Accuracy along with the value is stored in the dictionary scores.
    '''

    Cs = [0.001, 0.01, 0.1, 1.0, 10]
    scores = {}
    for c in Cs:
        svm = LinearSVC(C = c)
        start = time.time()
        svm.fit(Xtrain,ytrain)
        end = time.time()
        scores[c] = svm.score(Xval,yval)
        #print("SVM:",c,scores[c])
        #print("Time Liblinear:",c,end-start)
    
    '''
    The key-value pair corresponding to the maximum accuracy is selected, and used for training over the entire data.
    '''
    max_acc = 0
    max_key = None
    for k in scores.keys():
        if scores[k] > max_acc:
            max_key = k
            max_acc = scores[k]
    #print("Max-C:",max_key)
    svm = LinearSVC(C = max_key)
    start = time.time()
    svm.fit(X,y)
    end = time.time()
    ypred = svm.predict(Xtest)

    #print("Time Final SVM-Liblinear:",end-start)
    #print("SVM:Liblinear",svm.score(Xtest,ytest))
    write_predictions(output_file,ypred)

    '''SGD Regressor. The hyperparameter to be tuned is the regularization parameter alpha, which is defaulted to 0.0001
    Again the key (alpha) and the value (accuracy) is stored in a dictionary scores.'''

    # alphas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # scores = {}
    # for alph in alphas:
    #     start= time.time()
    #     svm_sgd = SGDClassifier(alpha = alph)
    #     svm_sgd.fit(Xtrain,ytrain)
    #     end = time.time()
    #     #print("Training Time SGD:",alph,end-start)
    #     scores[alph] = svm_sgd.score(Xval,yval) 
    #     #print("SVM:SGD:",alph,scores[alph])

    # '''
    # Hyper-parameter corresponding to the max-accuracy is selected and is used for training over the combined train-val
    # set and tested over the test set.
    # '''

    # max_acc = 0
    # max_key = None
    # for k in scores.keys():
    #     if scores[k] > max_acc:
    #         max_key = k
    #         max_acc = scores[k]
    # #print("Max-Alpha:",max_key)
    # svm_sgd = SGDClassifier(alpha = max_key)
    # start = time.time()
    # svm_sgd.fit(X,y)
    # end = time.time()
    # #print("Time Final SGD:",end-start)
    # #print("SVM:SGD",svm_sgd.score(Xtest,ytest))


    # Liblinear SVM with hinge loss

    # start = time.time()
    # Cs = [1e-3, 0.01, 0.1, 1.0, 10]
    # scores = {}
    # for c in Cs:
    #     svm = LinearSVC(C = c, loss='hinge', max_iter=10000)
    #     svm.fit(Xtrain,ytrain)
    #     scores[c] = svm.score(Xval,yval)
    #     print("SVM:",c,scores[c])
    #     end = time.time()
    #     print("Time:",end-start)
    # end = time.time()
    # print("CV Time:",end-start)

    # max_acc = 0
    # max_key = None
    # for k in scores.keys():
    #     if scores[k] > max_acc:
    #         max_key = k
    #         max_acc = scores[k]
    # #print("Max-Alpha:",max_key)
    # svm_sgd = SGDClassifier(alpha = max_key)
    # start = time.time()
    # svm_sgd.fit(X,y)
    # end = time.time()
    # #print("Time Final SGD:",end-start)
    # #print("SVM:SGD",svm_sgd.score(Xtest,ytest))

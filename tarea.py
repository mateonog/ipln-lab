import pandas
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

from sklearn.cross_validation import *
from scipy.stats import sem

import sys


def leer_csv ():
    corpus = pandas.read_csv("corpus_humor_training.csv",encoding='utf-8')
    corpus.drop(corpus[corpus['1']+corpus['2']+corpus['3']+corpus['4']+corpus['5']+corpus['n'] < 3].index, inplace=True)
    corpus['humoristico'] = np.where(corpus['1']+corpus['2']+corpus['3']+corpus['4']+corpus['5'] >= (corpus['n']+corpus['1']+corpus['2']+corpus['3']+corpus['4']+corpus['5'])/2,'si','no')
    corpus = corpus.drop(labels=['1','2','3','4','5','n'],  axis=1)
    return corpus

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score
    # method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print (scores)
    print (np.mean(scores))
    print (sem(scores))
    #print ("Mean score: {0:.3f} (+/-{1:.3f})").format(, sem(scores))

if __name__ == '__main__':
#    for texto in leer_csv()['text'][:20]:
#        print(texto + '\n')
    corpus = leer_csv()
    clf= Pipeline([('vect', CountVectorizer()),('clf', MultinomialNB()),])
    X_train, X_test, y_train, y_test = train_test_split(corpus.text, corpus.humoristico, test_size=0.20, random_state=33)
    evaluate_cross_validation(clf, corpus.text, corpus.humoristico, 5)
    clf.fit(corpus.text,corpus.humoristico)
    #clf.fit(X_train,y_train)
    asd =clf.predict([sys.argv[1]])
    print (clf.score(X_test, y_test))
    print ('¿Es gracioso? : ')
    print (asd[0])

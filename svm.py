import re

import pandas as pd
import numpy as np
import nltk
# import scikit-learn
import pyphen
from nltk.corpus import wordnet
import pyphen
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import math
from data.dale_chall import DALE_CHALL
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import GaussianNB


TRAIN_FILE = "data/train.xlsx"
TEST_FILE = "data/test.xlsx"
# TRAIN_FILE = "tests/train_sliced.xlsx"
# TEST_FILE = "tests/train_test_slice.xlsx"


SUBMISSION_FILE = "submissions/submission_svm.csv"

def nr_syllables(word):
    lang = pyphen.Pyphen(lang='en')
    x = lang.inserted(word, '-').split('-')
    return len(x)


def count_vowels(word):
    return sum(map(word.lower().count, "aeiou"))


def load_data(file):
    dtypes = {"sentence": str, "token": str, "complex": int}
    data_set = pd.read_excel(file, dtype=dtypes, keep_default_na=False)
    return data_set

def accuracy(preds, y):
    cfm = confusion_matrix(y, preds)
    return ((cfm[0][0] / (cfm[0][0] + cfm[0][1])) + (cfm[1][1] / (cfm[1][1] + cfm[1][0]))) / 2

def wordlist(file):
    train_data_set = load_data(file)
    wordlist_features = []
    n_meanings = set()
    for index in range(len(train_data_set.token)):

        is_name = 1 if \
            not train_data_set.sentence[index].index(train_data_set.token[index]) == 0 \
            and not train_data_set.token[index].isupper() \
            and train_data_set.token[index][0].isupper() \
            else 0

        wordlist_features.append([
                                    len(train_data_set.token[index]),
                                    int((lambda _word:  (train_data_set.token[index].lower() in DALE_CHALL)
                                                        or (train_data_set.token[index] in DALE_CHALL))
                                                        (train_data_set.token[index])),
                                    int((lambda _word: _word[0].isupper())(train_data_set.token[index])),
                                    float(nr_syllables(train_data_set.token[index])),
                                    int((lambda _word: _word.isupper())(train_data_set.token[index])),
                                    int((lambda _word: any(i.isdigit() for i in _word))(train_data_set.token[index])),
                                    # int((lambda _word: not bool(re.match("^[a-zA-Z0-9_]*$", _word)))(
                                    #   train_data_set.token[index])),
                                    # is_name,
                                    # count_vowels(train_data_set.token[index]),
                                    len(wordnet.synsets(train_data_set.token[index])),
                                    len(train_data_set.corpus[index])])

        n_meanings.add(x for x in wordnet.synsets(train_data_set.token[index]))
    #
    for index in range(len(train_data_set.token)):
        wordlist_features[index].append(list(train_data_set.token).count(train_data_set.token[index]))                  # self_frequency
    #     wordlist_features[index].append(len(wordnet.synsets(train_data_set.token[index])) / len(n_meanings))            #?? n_meanings / len(wordlist)
    #     wordlist_features[index].append(float(list(train_data_set.token).count(train_data_set.token[index]) / len(              #frequency in text
    #         train_data_set.token)))
        wordlist_features[index].append((lambda x: 1 if x > 0.004 else 0)(wordlist_features[index][len(wordlist_features[index])-1]))            #is_frequent

    y_train_set = []
    try:
        for cpx in train_data_set.complex:
            y_train_set.append(cpx)
    except Exception as e:
        print(e)
        pass

    return wordlist_features, y_train_set, train_data_set.index


train_data, y_train, train_index = wordlist(TRAIN_FILE)
test_data, y_test, test_index =  wordlist(TEST_FILE)


svc = SVC(kernel='rbf')
svc.fit(train_data, y_train)

preds = svc.predict(test_data)

try:
    print(accuracy(y_test, preds))
except ValueError:
    pass


df = pd.DataFrame()
df['id'] = test_index + 7663
df['complex'] = preds

df.to_csv(SUBMISSION_FILE, index=False)

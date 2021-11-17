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

from data.dale_chall import DALE_CHALL
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import GaussianNB

# nltk.download('punkt')
# nltk.download('wordnet')

TRAIN_FILE = "data/train.xlsx"
TEST_FILE = "data/test.xlsx"

#
# TRAIN_FILE = "tests/train_sliced.xlsx"
# TEST_FILE = "tests/train_test_slice.xlsx"


class Word:

    def __init__(self, corpus, length, is_dale, is_capitalized, n_syl, freq, freq_in_text, all_caps,
                 contains_numbers,
                 contains_specials, is_first, n_vowels, n_meanings, is_name):
        self.corpus = corpus
        self.length = length
        self.is_dale = is_dale
        self.is_capitalized = is_capitalized
        self.n_syl = n_syl
        self.freq = freq
        self.freq_in_text = float(freq_in_text)
        self.all_caps = all_caps
        self.contains_numbers = contains_numbers
        self.contains_specials = contains_specials
        self.is_first = is_first
        self.n_vowels = n_vowels
        self.is_long = 1 if self.length > 10 else 0
        self.short_not_dale = int((1 if self.length < 4 else 0) and not is_dale)
        self.is_frequent = 1 if self.freq_in_text > 0.0005 else 0
        self.n_meanings = n_meanings
        self.is_name = is_name

    def get_features(self):
        return list(
            [self.corpus, self.length, self.is_dale, self.is_capitalized, self.n_syl, self.freq, self.freq_in_text,
             self.all_caps, self.contains_numbers, self.contains_specials, self.is_first, self.n_vowels, self.is_long,
             self.short_not_dale, self.is_frequent, self.n_meanings, self.is_name])


class InputData:

    def __init__(self, data_file, is_test):
        self.data = self.load_data(data_file)
        self.is_test = is_test
        self.words = self.word_dict()

    @staticmethod
    def load_data(file):
        dtypes = {"sentence": str, "token": str, "complex": int}
        data_set = pd.read_excel(file, dtype=dtypes, keep_default_na=False)
        return data_set

    @staticmethod
    def nr_syllables(word):
        lang = pyphen.Pyphen(lang='en')
        x = lang.inserted(word, '-').split('-')
        return len(x)

    def word_dict(self):

        if self.data is None:
            raise Exception("You need to load some data first")

        words = dict({})
        for index in range(len(self.data)):
            # word = word_tokenize(self.data.token[index])
            word = self.data.token[index]
            # for word in target:
            if word not in "!?,.';[]{}@#$%^&*()``:":
                try:  # _TODO: update just freq
                    words.update({word: Word(len(self.data.corpus[index]),
                                             len(word),
                                             words[word].is_dale,
                                             words[word].is_capitalized,
                                             words[word].n_syl,
                                             words[word].freq + 1,
                                             (words[word].freq + 1) / len(self.data.token),  # (len(words) + 1),
                                             words[word].all_caps,
                                             words[word].contains_numbers,
                                             words[word].contains_specials,
                                             words[word].is_first,
                                             words[word].n_vowels,
                                             words[word].n_meanings,
                                             words[word].is_name
                                             )
                                  })
                    # print(word + ": in try")
                except KeyError:
                    words.update({word: Word(len(self.data.corpus[index]),
                                             len(word),
                                             int((lambda _word: word.lower() in DALE_CHALL)(word)),
                                             int((lambda _word: _word[0].isupper())(word)),
                                             self.nr_syllables(word),
                                             1,
                                             1 / len(self.data.token),  # (len(words) + 1),
                                             int((lambda _word: _word.isupper())(word)),
                                             int((lambda _word: any(i.isdigit() for i in _word))(word)),
                                             int((lambda _word: not bool(re.match("^[a-zA-Z0-9_]*$", _word)))(
                                                 word)),
                                             int((lambda _word: True if self.data.sentence[index].index(
                                                 _word) == 0 else False)(word)),
                                             self.count_vowels(word),
                                             len(wordnet.synsets(word)),
                                             self.is_name(self.data.token[index], self.data.sentence[index])
                                             )
                                  })
                    # print(word + ":[new value: " + str(ex) + " detected and added to dict]")
        return words

    @staticmethod
    def count_vowels(word):
        return sum(map(word.lower().count, "aeiou"))

    @staticmethod
    def is_name(word, sentence):
        return 1 if \
            not sentence.index(word) == 0 \
            and not word.isupper() \
            and word.isupper() \
            else 0


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


def wordlist(file):
    train_data = load_data(file)
    wordlist_features = []
    for index in range(len(train_data.token)):
        is_name = 1 if \
            not train_data.sentence[index].index(train_data.token[index]) == 0 \
            and not train_data.token[index].isupper() \
            and train_data.token[index][0].isupper() \
            else 0

        wordlist_features.append([len(train_data.corpus[index]),                                                        #encoded corpus
                                  len(train_data.token[index]),                                                         #length
                                  int((lambda _word: train_data.token[index].lower() in DALE_CHALL)(                    #is_dale
                                      train_data.token[index])),
                                  int((lambda _word: _word[0].isupper())(train_data.token[index])),                     #is_capitalised
                                  #nr_syllables(train_data.token[index]),                                                #n_syllables
                                  int((lambda _word: _word.isupper())(train_data.token[index]))  ,                       #caps
                                  # int((lambda _word: any(i.isdigit() for i in _word))(train_data.token[index]))])        #contains_numbers
                                  #int((lambda _word: not bool(re.match("^[a-zA-Z0-9_]*$", _word)))(                     #contains_specials
                                  #     train_data.token[index]))
                                  #is_name,                                                                              #is_name
                                  #count_vowels(train_data.token[index]),                                                #n_vowels
                                  len(wordnet.synsets(train_data.token[index]))  ])                                    #n_meanings

    # for index in range(len(train_data.token)):
        wordlist_features[index].append(list(train_data.token).count(train_data.token[index]))                          #self_frequency
        wordlist_features[index].append(wordlist_features[index][len(wordlist_features[index])-2] / len(wordlist_features))
        # wordlist_features[index].append(float(list(train_data.token).count(train_data.token[index]) / len(              #frequency in text
        #     train_data.token)))
        #wordlist_features[index].append((lambda x: 1 if x > 0.004 else 0)(wordlist_features[index][len(wordlist_features[index])-1]))            #is_frequent

    y_train = []
    try:
        for cpx in train_data.complex:
            y_train.append(cpx)
    except:
        pass



    return wordlist_features, y_train


def knn(n):
    train_data, y_train = wordlist(TRAIN_FILE)
    test_data, _ = wordlist(TEST_FILE)
    test_data_dict = InputData(TEST_FILE, is_test=False)

    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(train_data, y_train)
    preds = model.predict(test_data)

    df = pd.DataFrame()
    df['id'] = test_data_dict.data.index + 7663
    df['complex'] = preds
    df.to_csv("submissions/submission_knn.csv", index=False)

    return preds

def svm():
    train_data, y_train = wordlist(TRAIN_FILE)
    test_data, _ = wordlist(TEST_FILE)

    svc = SVC(kernel='linear')
    svc.fit(train_data, y_train)

    preds = svc.predict(test_data)

    df = pd.DataFrame()
    df['id'] = 1  # test_data_dict.data.index + 7663
    df['complex'] = preds
    df.to_csv("submissions/submission_svm.csv", index=False)

def knn_dict():
    TRAIN_FILE = "data/train.xlsx"
    TEST_FILE = "data/test.xlsx"

    train_data_dict = InputData(TRAIN_FILE, is_test=False)
    test_data_dict = InputData(TEST_FILE, is_test=True)

    X_train = []
    y_train = []
    for token in train_data_dict.data.token:
        X_train.append(train_data_dict.data.words[token].get_features())

    X_test = []
    for token in test_data_dict.data.token:
        X_test.append(test_data_dict.words[token].get_features())

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    df = pd.DataFrame()
    df['id'] = 1  # test_data_dict.data.index + 7663
    df['complex'] = preds
    df.to_csv("submissions/submission_knn.csv", index=False)

def svm_dict():
    TRAIN_FILE = "data/train.xlsx"
    TEST_FILE = "data/test.xlsx"

    train_data_dict = InputData(TRAIN_FILE, is_test=False)
    test_data_dict = InputData(TEST_FILE, is_test=True)

    X_train = []
    y_train = []
    for token in train_data_dict.data.token:
        X_train.append(train_data_dict.words[token].get_features())

    X_test = []
    for token in test_data_dict.data.token:
        X_test.append(test_data_dict.words[token].get_features())

    svc = svm.SVC(kernel='linear', C=1, gamma=5)
    svc.fit(X_train, y_train)

    preds = svc.predict(X_test)

    df = pd.DataFrame()
    df['id'] = 1  # test_data_dict.data.index + 7663
    df['complex'] = preds
    df.to_csv("submissions/submission_knn.csv", index=False)

def accuracy(preds, y):
    cfm = confusion_matrix(preds, y)
    return ((cfm[0][0] / (cfm[0][0] + cfm[0][1])) + (cfm[1][1] / (cfm[1][1] + cfm[1][0]))) / 2




if __name__ == '__main__':

    #
    #svm()
    # # knn_dict()
    # # svm_dict()
    # train_data_dict = InputData(TRAIN_FILE, is_test=False)
    test_data_dict = InputData(TEST_FILE, is_test=True)
    #
    # X_train = []
    # y_train = []
    #
    # for token in train_data_dict.data.token:
    #     X_train.append(train_data_dict.words[token].get_features())
    #
    # for cpx in train_data_dict.data.complex:
    #     y_train.append(cpx)
    #

    # for k in range(1,5,1):
    #     y_test_train = []
    #     for cpx in test_data_dict.data.complex:
    #         y_test_train.append(cpx)
    #
    #     preds = knn(k)
    #     print(y_test_train)
    #     print(list(preds))
    #     print("k = " + str(k))
    #     f = open('acc.txt', 'r')
    #     old_acc = f.readline()
    #     if old_acc != 'nan' and old_acc != '':
    #         print("{:.4f}".format(float(old_acc)) + " vs " + "{:.4f}".format(accuracy(preds, y_test_train)))
    #     else:
    #         print('No previous value registered. Acc: ' + "{:.4f}".format(accuracy(preds, y_test_train)))
    #     f = open('acc.txt', 'w')
    #     f.write("{:.4f}".format(accuracy(preds, y_test_train)))

    train_data, y_train = wordlist(TRAIN_FILE)
    test_data, _ = wordlist(TEST_FILE)

    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(train_data, y_train)

    # Predict the response for test dataset
    y_pred = gnb.predict(test_data)

    #print(accuracy(y_pred, y_train))

    df = pd.DataFrame()
    df['id'] = 1#test_data_dict.data.index + 7663
    df['complex'] = y_pred
    df.to_csv("submissions/submission_nb.csv", index=False)

    # X_test = []
    # y_test = []
    # for token in test_data_dict.data.token:
    #     X_test.append(test_data_dict.words[token].get_features())
    #
    # model = KNeighborsClassifier(n_neighbors=3)
    # model.fit(X_train, y_train)
    # preds = model.predict(X_test)
    #
    # df = pd.DataFrame()
    # df['id'] = 2  # test_data_dict.data.index + 7663
    # df['complex'] = preds
    # df.to_csv("submissions/submission_knn_dict.csv", index=False)
    #
    # train_data_dict = InputData(TRAIN_FILE, is_test=False)
    # test_data_dict = InputData(TEST_FILE, is_test=True)
    #
    # X_train = []
    # y_train = []
    #
    # for token in train_data_dict.data.token:
    #     X_train.append(train_data_dict.words[token].get_features())
    #
    # for cpx in train_data_dict.data.complex:
    #     y_train.append(cpx)
    #
    # X_test = []
    # y_test = []
    # for token in test_data_dict.data.token:
    #     X_test.append(test_data_dict.words[token].get_features())
    #
    # model = KNeighborsClassifier(n_neighbors=3)
    # model.fit(X_train, y_train)
    # preds = model.predict(X_test)
    #
    # svc = SVC(kernel='linear', C=1, gamma=5)
    # svc.fit(X_train, y_train)
    #
    # preds = svc.predict(X_test)
    #
    # df = pd.DataFrame()
    # df['id'] = 2  # test_data_dict.data.index + 7663
    # df['complex'] = preds
    # df.to_csv("submissions/submission_svm_dict.csv", index=False)



import re

import numpy as np
import pandas as pd
from nltk.corpus import wordnet
import pyphen
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from data.dale_chall import DALE_CHALL
from sklearn.naive_bayes import GaussianNB
from datetime import datetime

TRAIN_FILE = "data/train.xlsx"
TEST_FILE = "data/test.xlsx"
# TRAIN_FILE = "tests/train_sliced.xlsx"
# TEST_FILE = "tests/train_test_slice.xlsx"

SUBMISSION_FILE = "submissions/submission_nb.csv"


def nr_syllables(word):
    lang = pyphen.Pyphen(lang='en')
    x = lang.inserted(word, '-').split('-')
    return len(x)


def count_vowels(word):
    return len(word) - sum(map(word.lower().count, "aeiou"))


def load_data(file):
    dtypes = {"sentence": str, "token": str, "complex": int}
    data_set = pd.read_excel(file, dtype=dtypes, keep_default_na=False)
    return data_set


def accuracy(preds, y):
    cfm = confusion_matrix(y, preds)
    return ((cfm[0][0] / (cfm[0][0] + cfm[1][0])) + (cfm[1][1] / (cfm[1][1] + cfm[0][1]))) / 2


def wordlist(file):
    train_data_set = load_data(file)
    print("[" + str(datetime.now().time())[:-4] + "] Generating features")
    wordlist_features = []
    n_meanings = set()
    text = list()
    text += ' '
    for index in range(len(train_data_set.token)):
        is_name = 1 if \
            not train_data_set.sentence[index].index(train_data_set.token[index]) == 0 \
            and not train_data_set.token[index].isupper() \
            and train_data_set.token[index][0].isupper() \
            and not train_data_set.token[index].lower() in DALE_CHALL \
            else 0
        text[0] += train_data_set.sentence[index] + " "

        word = train_data_set.token[index]

        wordlist_features.append([float(len(word)),
                                  int((lambda _word: _word.lower() in DALE_CHALL or _word in DALE_CHALL)(word)),        #check in dale wordlist
                                  # int((lambda _word: _word[0].isupper())(word)),                                        #check capitalised
                                  float(nr_syllables(word)),                                                            #check nr_syllables
                                  # int((lambda _word: _word.isupper())(word)),                                           #check if whole word is caps
                                  int((lambda _word: any(char.isdigit() for char in _word))(word)),                           #check if word contains/is number
                                  int((lambda _word: not bool(re.match("^[a-zA-Z_]*$", _word)))(word)),                 #check if word contains special chars
                                  is_name,                                                                              #check if word is name (inside sentence+capitalised)
                                  # count_vowels(word),                                                                    #count no vowels
                                  len(wordnet.synsets(word)),                                                           #count how many meanings the word has
                                  len(train_data_set.corpus[index])                                                     #encoded corpus of word
                                  ])

        n_meanings.add(x for x in wordnet.synsets(train_data_set.token[index]))                                         # create unique meanings

    n_unq_words = len(set(text[0]))
    n_words = len(text[0].split())
    n_unq_meanings = len(set(n_meanings))
    train_data_token_list = list(train_data_set.token)
    print(n_words, "words", n_unq_words, "unique")
    print(n_unq_words / n_words, "average frequency per word")

    for index in range(len(train_data_set.token)):
        wordlist_features[index].append(train_data_token_list.count(train_data_set.token[index]))                       #calculate self_frequency
        wordlist_features[index].append(wordlist_features[index][len(wordlist_features[index])-3]/ n_unq_meanings)      #avg meanings per word??
        freq = float(text[0].count(train_data_set.token[index]) / n_words)                                              #global frequency of word
        wordlist_features[index].append(freq)
        wordlist_features[index].append((lambda x: 1 if x >= n_unq_words / n_words else 0)(freq))                       #determine if the word is frequent 0.0007476 avg on train_data

    y_train_set = []
    try:                                                                                                                #test cases have no y column
        for cpx in train_data_set.complex:
            y_train_set.append(cpx)
    except AttributeError as e:
        print(e)
        pass

    return wordlist_features, y_train_set, train_data_set.index

print("[" + str(datetime.now().time())[:-4] + "] Loading train data")
train_data, y_train, train_index = wordlist(TRAIN_FILE)
print("[" + str(datetime.now().time())[:-4] + "] Loading test data")
test_data, y_test, test_index = wordlist(TEST_FILE)

ac = []

kf = KFold(n_splits=10, random_state=None, shuffle=True)
gnb = GaussianNB()

print("[" + str(datetime.now().time())[:-4] + "] Starting KFold")
for xtrain_index, xtest_index in kf.split(np.array(train_data)):
    X_train = (np.array(train_data))[xtrain_index]
    X_test = (np.array(train_data))[xtest_index]
    xy_train = (np.array(y_train))[xtrain_index]
    xy_test = (np.array(y_train))[xtest_index]
    gnb.fit(X_train, xy_train)
    xy_pred = gnb.predict(X_test)
    ac.append(accuracy(xy_test, xy_pred))

av_ac = sum(ac) / 10
print("KFold accuracy:", av_ac)

print("[" + str(datetime.now().time())[:-4] + "] Starting train test")
gnb.fit(train_data, y_train)

print("[" + str(datetime.now().time())[:-4] + "] Predicting over train data")
y_pred = gnb.predict(train_data)

print("Test accuracy:", accuracy(y_train, y_pred))

print("[" + str(datetime.now().time())[:-4] + "] Starting main test")
gnb.fit(train_data, y_train)
print("[" + str(datetime.now().time())[:-4] + "] Predicting test data")
y_pred = gnb.predict(test_data)

#for testing on sliced train data
try:
    print(accuracy(y_test, y_pred))
except ValueError:
    pass

print("[" + str(datetime.now().time())[:-4] + "] Writing to csv")
df = pd.DataFrame()
df['id'] = test_index + 7663
df['complex'] = y_pred

df.to_csv(SUBMISSION_FILE, index=False)
print("[" + str(datetime.now().time())[:-4] + "] Finished successfully")
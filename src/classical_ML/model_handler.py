from src.classical_ML.src.config import *
from src.classical_ML.src.data_reader import DataUnification, LoadEssaysSentences
from src.classical_ML.src.vectorizer import Vectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedKFold
import os.path
import numpy as np

data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'

# DataUnification()
with open(data_dir + 'essays_sentences.json', encoding='utf-8') as f:
    sentences_all = json.load(f)

y = [sent['sent-class'] for sent in sentences_all]

tmp = []
for c in y:
    if c == 'c':
        tmp.append(1)
    elif c == 'p':
        tmp.append(1)
    else:
        tmp.append(0)
y = tmp

print('end parsing')

## Uncomment this if this is first time
vectorizer = Vectorizer()
vectorizer.fit(sentences_all)


def train_svm():
    '''
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    for train_idx, test_idx in kf.split(sentences_all, y):
        train_sentences = sentences_all[train_idx[0] : train_idx[-1] + 1]
        test_sentences = sentences_all[test_idx[0] : test_idx[-1] + 1]
        print('Start vectorization')
        y_train = [sent['sent-class']  for sent in train_sentences]
        y_test = [sent['sent-class']  for sent in test_sentences]
        x_vec_train = vectorizer.transform(train_sentences)
        x_vec_test = vectorizer.transform(test_sentences)
        print('End vectorization')
        svmClf = svm.SVC(kernel='rbf', C=107)
        svmClf.fit(x_vec_train, y_train)
        y_pred_svm = svmClf.predict_proba(x_vec_test)
        svm_acc = accuracy_score(y_test, y_pred_svm)
        print(f'Accuracy score on testing data: {svm_acc}')
    '''

    train_sentences = [sent for sent in sentences_all if sent['train'] and sent['sent-class'] != 'n']

    test_sentences = [sent for sent in sentences_all if not sent['train'] and sent['sent-class'] != 'n']

    print('Start vectorization')
    ## General dataset 
    ## y_train = [sent['sent-class']  for sent in train_sentences]
    ## y_test = [sent['sent-class']  for sent in test_sentences]
    ## arg and non-arg dataset
    ## y_train = [0 if sent['sent-class'] == 'n' else 1 for sent in train_sentences]
    ## y_test = [0 if sent['sent-class'] == 'n' else 1 for sent in test_sentences]
    ## claim and premise dataset 
    y_train = [0 if sent['sent-class'] == 'c' else 1 for sent in train_sentences]
    y_test = [0 if sent['sent-class'] == 'c' else 1 for sent in test_sentences]
    x_vec_train = vectorizer.transform(train_sentences)
    x_vec_test = vectorizer.transform(test_sentences)
    print('End vectorization')
    if not os.path.isfile('svm_model_2cs_c_p.sav'):
        svmClf = svm.SVC(kernel='rbf', C=107)
        scores = cross_val_score(svmClf, x_vec_train, y_train, cv=2, n_jobs=36)
        ##print(f'10-fold cross validation accuracy score: {scores.mean()}')
        svmClf.fit(x_vec_train, y_train)

        # Save the model to disk
        filename = 'svm_model_2cs_c_p.sav'
        pickle.dump(svmClf, open(filename, 'wb'))
        print("Model saved successfully.")

        y_pred_svm = svmClf.predict(x_vec_test)

        ##f1 = f1_score(x_vec_test, y_pred_svm, average='weighted')
        ##print(f'F1 score on testing data: {f1}')

        print(classification_report(y_test, y_pred_svm, digits=4))

    else:
        # Load the model from disk
        svmClf = pickle.load(open('svm_model_2cs_c_p.sav', 'rb'))
        print("Model loaded successfully.")

        y_pred_svm = svmClf.predict(x_vec_test)
        print(classification_report(y_test, y_pred_svm, digits=4))


def test_double_svm():
    test_sentences = [sent for sent in sentences_all if not sent['train']]

    print('Start vectorization')
    ## General dataset 
    ## y_train = [sent['sent-class']  for sent in train_sentences]
    ## y_test = [sent['sent-class']  for sent in test_sentences]
    ## arg and non-arg dataset
    ## y_train = [0 if sent['sent-class'] == 'n' else 1 for sent in train_sentences]
    ## y_test = [0 if sent['sent-class'] == 'n' else 1 for sent in test_sentences]
    ## claim and premise dataset 
    y_test = [sent['sent-class'] for sent in test_sentences]
    x_vec_test = vectorizer.transform(test_sentences)
    y_pred_svm_1 = []
    y_pred_svm_2 = []
    print('End vectorization')
    if not os.path.isfile('svm_model_2cs_arg_non.sav'):
        print("ERROR! Model not found.")

    else:
        # Load the model from disk
        svmClf = pickle.load(open('svm_model_2cs_arg_non.sav', 'rb'))
        print("Model 1 SVM loaded successfully.")

        y_pred_svm_1 = svmClf.predict(x_vec_test)

    if not os.path.isfile('svm_model_2cs_c_p.sav'):
        print("ERROR! Model not found.")

    else:
        # Load the model from disk
        svmClf = pickle.load(open('svm_model_2cs_c_p.sav', 'rb'))
        print("Model 2 SVM loaded successfully.")

        y_pred_svm_2 = svmClf.predict(x_vec_test)

    y_pred_svm = []
    for pred_1, pred_2 in zip(y_pred_svm_1, y_pred_svm_2):
        if pred_1 == 0:
            y_pred_svm.append('n')
        else:
            if pred_2 == 0:
                y_pred_svm.append('c')
            else:
                y_pred_svm.append('p')

    print(classification_report(y_test, y_pred_svm, digits=4))


# for second stage
def svm_test_second_stage(test_sentences):
    print('Stage two SVM testing')

    print('Start vectorization')
    x_vec_test = vectorizer.transform(test_sentences)
    print('End vectorization')
    # Load the model from disk
    svmClf = pickle.load(open('svm_model_2cs_c_p.sav', 'rb'))
    print("Model 2 SVM loaded successfully.")

    y_pred_svm = svmClf.predict(x_vec_test)
    return y_pred_svm.tolist()


# for first stage
def svm_test_first_stage(test_sentences):
    print('Stage first SVM testing')

    print('Start vectorization')
    x_vec_test = vectorizer.transform(test_sentences)
    print('End vectorization')
    # Load the model from disk
    svmClf = pickle.load(open('models\\svm_model_2cs_arg_non.sav', 'rb'))
    print("Model 1 SVM loaded successfully.")

    y_pred_svm = svmClf.predict(x_vec_test)
    return y_pred_svm.tolist()


def svm_train_and_predict(train_sentences, y_train, test_sentences):
    svmClf = svm.SVC(kernel='rbf', C=107, probability=True)
    print('Start vectorization')
    # y_train = [sent['sent-class']  for sent in train_sentences]
    y_test = [sent['sent-class'] for sent in test_sentences]
    x_vec_train = vectorizer.transform(train_sentences)
    x_vec_test = vectorizer.transform(test_sentences)
    print('End vectorization')
    svmClf.fit(x_vec_train, y_train)
    y_pred_svm = svmClf.predict_proba(x_vec_test)
    # svm_acc = accuracy_score(y_test, y_pred_svm)
    # print(f'Accuracy score on testing data: {svm_acc}')
    return svmClf, y_pred_svm.tolist()


def svm_predict(model, sentences, proba=True):
    x_vec = vectorizer.transform(sentences)
    if proba:
        preds = model.predict_proba(x_vec)
        return preds.tolist()

    preds = model.predict(x_vec)
    return preds.tolist()

import nltk
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from unidecode import unidecode
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, plot_roc_curve
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import auc
import codecs


path = 'C:/Users/gabis/Documents/tcc/Fake_News_Ckassifiers/input/validation.csv'
folder = 'C:/Users/gabis/Documents/tcc/Fake_News_Ckassifiers/input/dataToProcess'
GLOVE_DIR = 'C:/Users/gabis/PycharmProjects/Glove/input/glove_s50.txt'

def preProcess():
    #metodo utilizado apenas uma vez para pre processamento da base
    stop_words = nltk.corpus.stopwords.words("portuguese")
    folder
    for file in os.listdir(folder):
        word_list = open(folder + file,encoding='utf-8')
        line = word_list.read()
        words = line.split()
        for r in words:
            if not r in stop_words:
                r = re.sub(r'[^\w\s]', ' ', r)
                r = re.sub("\d+", ' ', r)
                appendFile = open('validation.csv', 'a', encoding="utf-8")
                appendFile.write(" " + unidecode(r.lower()))
                appendFile.close()
        appendFile = open('validation.csv', 'a',encoding='utf-8')
        appendFile.write(";REAL\n")

def preProcess2():
    #metodo utilizado apenas uma vez para pre processamento da base

    file1 = codecs.open(path, 'r', 'utf-8')
    i = 0
    for line in file1.readlines():
       appendFile = open( 'path' + i.__str__() +'.csv', 'a', encoding='utf-8')
       appendFile.write(line)
       appendFile.close()
       i= i+1


def ramdomizeBase():
    # metodo utilizado apenas uma vez para embaralhar a base
    destin_folder = 'C:/Users/gabis/Documents/tcc/Fake_News_Ckassifiers/input/validation2.csv'
    with open(path, 'r', encoding='utf-8') as r, open(destin_folder, 'w', encoding='utf-8') as w:
        data = r.readlines()
        header, rows = data[0], data[1:]
        random.shuffle(rows)
        rows = '\n'.join([row.strip() for row in rows])
        w.write(header + rows)


def get_base(algorithm):
    #metodo utilizado dentro dos algoritimos para utilizar a base de noticias
    df = pd.read_csv(path)
    if algorithm == "Glove":
        df.head()
        df.info()
        x = df['text']
        y = pd.get_dummies(df['label'])
        no_of_fakes = df.loc[df['label'] == 'FAKE'].count()[0]
        no_of_trues = df.loc[df['label'] == 'REAL'].count()[0]
        y = np.array(y)
        return x, y

    else:
        df.head()
        df = df.fillna('')
        df['text'] = df['text']
        df.head()
        df = df[df['label'] != '']
        df.loc[df['label'] == 'REAL', 'label'] = 'REAL'
        df.loc[df['label'] == 'FAKE', 'label'] = 'FAKE'

        no_of_fakes = df.loc[df['label'] == 'FAKE'].count()[0]
        no_of_trues = df.loc[df['label'] == 'REAL'].count()[0]

        print(df['label'].unique())
        print(no_of_fakes)
        print(no_of_trues)

        y = df['label'].values
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(df['text'].values)
        print(x)
        x = x.toarray()
        return x, y



def naive_bayes():
    #metodo simples de implementação de naive bayes
    x, y = get_base("Naive")
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.20, random_state=11)

    model = MultinomialNB()

    model_NB = model.fit(x_train, y_train)
    svc_pred = model_NB.predict(x_test)

    print("Acuracia do Naive Classifier: {}%".format(round(accuracy_score(y_test, svc_pred) * 100, 2)))
    print("\nMatrix de confusão Naive:\n")
    print(confusion_matrix(y_test, svc_pred))
    print("\nDetalhes da classificação Naive :\n")
    print(classification_report(y_test, svc_pred))

def svm():
    #metodo simples de implementação de svm
    x, y = get_base("SVM")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
    model = MultinomialNB()

    model_svc = model.fit(y_test, y_train)
    svc_pred = model_svc.predict(x_test)
    print("Accuracy do SVM Classifier: {}%".format(round(accuracy_score(y_test, svc_pred) * 100, 2)))
    print("\nMatrix de confusão SVM:\n")
    print(confusion_matrix(y_test, svc_pred))
    print("\nDetalhes da classificação SVM :\n")
    print(classification_report(y_test, svc_pred))

def naive_bayesCross():
    # metodo de implementação de naive bayes em K-fold
    model = MultinomialNB()
    k_fold_cross("Naive", model)

def svmCross():
    # metodo de implementação de svm em K-fold
    model = LinearSVC()
    k_fold_cross("SVM", model)

def k_fold_cross(algorithm,model):
    #metodo do kFold

    acc_score = []
    X, y = get_base(algorithm)
    k = 10
    kf = KFold(n_splits=k, random_state=None)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)
        acc = accuracy_score(y_test,pred_values)
        acc_score.append(acc)
        print("\nMatrix de confusão SVM:\n")
        print(confusion_matrix(y_test, pred_values))

    avg_acc_score = sum(acc_score) / k

    print('Acuracia de cada fold - {}'.format(acc_score))
    print('Acuracia media : {}'.format(avg_acc_score))



def glove():
    #metodo para implementação do glove
    x, y = get_base("Glove")
    MAX_NB_WORDS = 100000
    MAX_SEQUENCE_LENGTH = 1000
    VALIDATION_SPLIT = 0.2
    EMBEDDING_DIM = 50

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(x)
    sequences = tokenizer.texts_to_sequences(x)
    word_index = tokenizer.word_index
    print('Tamanho do vocabulario:', len(word_index))

    data = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)


    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = y[indices]

    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[: -num_validation_samples]
    y_train = labels[: -num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    print('Numeros em categorias:')
    print('treino: ', y_train.sum(axis=0))
    print('validacao: ', y_val.sum(axis=0))

    embeddings_index = {}
    f = open(GLOVE_DIR, encoding='utf8')
    for line in f:
        try:
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype='float32')
        except:
            print()
    f.close()
    print("Processando embending matrix", end="")
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("Finalizado!")

    model = Sequential()
    model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
    model.add(Embedding(len(word_index) + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False,
                        name='embeddings'))
    model.add(LSTM(60, return_sequences=True, name='lstm_layer'))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation="sigmoid"))

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy',
                              tf.keras.metrics.AUC(),
                              tf.keras.metrics.TruePositives(name='tp'),
                              tf.keras.metrics.FalsePositives(name='fp'),
                              tf.keras.metrics.TrueNegatives(name='tn'),
                              tf.keras.metrics.FalseNegatives(name='fn'),
                              tf.keras.metrics.Recall()])


    history = model.fit(x_train, y_train, epochs=10, batch_size=128,  validation_data=(x_val, y_val))
    result = model.evaluate(x_val, y_val)
    print(pd.DataFrame(history.history))
    accuracy = result[1]
    print(pd.DataFrame(result))
    f1_score = 2 * (accuracy * result[7]) / (accuracy + result[7])
    print("f1 score:" + f1_score.__str__())
    print(f"[+] Accuracy: {accuracy * 100:.2f}%")



def naive_bayes_all_validations():
    #metodo com k-fold e plot de curva roc do algoritimo naive bayes
    draw_cv_roc_curve(MultinomialNB())

def svm_all_validations():
    #metodo com k-fold e plot de curva roc do algoritimo svm
    draw_cv_roc_curve(LinearSVC())

def draw_cv_roc_curve(model):

    #metodo com k-fold e plot de curva roc

    X, y = get_base("")
    classifier = model


    k = 10
    cv = StratifiedKFold(n_splits=k)
    acc_score = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        print(mean_fpr)
        print(viz.fpr)
        print(viz.tpr)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        pred_values = classifier.predict(X[test])
        acc = accuracy_score(y[test], pred_values)
        acc_score.append(acc)
        print("fold {}".format(i))
        print("Matriz de confusao")
        print(confusion_matrix(y[test], pred_values))
        print("Report")
        print(classification_report(y[test], pred_values))



    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'ROC Média (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Característica de Operação do Receptor")
    ax.legend(loc="lower right")

    avg_acc_score = sum(acc_score) / k
    print('Acurácia de cada fold  - {}'.format(acc_score))
    print('Acurácia Média : {}'.format(avg_acc_score))
    plt.show()

import nltk
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import tensorflow as tf
from unidecode import unidecode
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_curve, \
    average_precision_score, roc_curve, plot_roc_curve
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import auc


def preProcess():
    stop_words = nltk.corpus.stopwords.words("portuguese")
    path = "C:/Fake.br-Corpus/full_texts/filtradas/teste/"
    for file in os.listdir(path):
        word_list = open(path+file, encoding="utf-8")
        line = word_list.read()
        words = line.split()
        for r in words:

            if not r in stop_words:
                r = re.sub(r'[^\w\s]', ' ', r)
                r = re.sub("\d+", ' ', r)
                appendFile = open('processadas.csv', 'a', encoding="utf-8")
                appendFile.write(" " + unidecode(r.lower()))
                appendFile.close()
        appendFile = open('processadas.csv', 'a', encoding="utf-8")
        appendFile.write(";true\n")
        appendFile.close()


def get_base(algorithm):
    df = pd.read_csv('C:/Users/gabis/PycharmProjects/Glove/input/processadas.csv')

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
        x = x.toarray()
        return x, y



def glove():
    x, y = get_base("Glove")
    MAX_NB_WORDS = 100000  # maximo de palavras tokenrizadas
    MAX_SEQUENCE_LENGTH = 1000  # max length of each sentences, including padding
    VALIDATION_SPLIT = 0.2  # 20% of data for validation (not used in training)
    EMBEDDING_DIM = 50  # embedding dimensions for word vectors
    GLOVE_DIR = "C:/Users/gabis/PycharmProjects/Glove/input/glove_s50.txt"

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(x)
    sequences = tokenizer.texts_to_sequences(x)
    word_index = tokenizer.word_index
    print('Vocabulary size:', len(word_index))

    data = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', y.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = y[indices]

    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[: -num_validation_samples]
    y_train = labels[: -num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    print('Number of entries in each category:')
    print('training: ', y_train.sum(axis=0))
    print('validation: ', y_val.sum(axis=0))

    print('Tokenized sentences: \n', data[10])
    print('One hot label: \n', labels[10])

    embeddings_index = {}
    f = open(GLOVE_DIR, encoding='utf8')
    print('Loading Glove from:', GLOVE_DIR, '…', end='')
    for line in f:
        try:
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype='float32')
        except:
            print()
    f.close()
    print("Done.\n Proceeding with Embedding Matrix…", end="")
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("Completed!")

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
                  metrics=['accuracy',tf.keras.metrics.AUC(from_logits=True)])

    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))

    random_num = np.random.randint(0, 100)
    test_data = x[random_num]
    test_label = y[random_num]
    clean_test_data = test_data
    test_tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    test_tokenizer.fit_on_texts(clean_test_data)
    test_sequences = tokenizer.texts_to_sequences(clean_test_data)
    word_index = test_tokenizer.word_index
    test_data_padded = pad_sequences(test_sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    print('Sample data:', x[0], y[0])

def naive_bayes():
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

    x, y = get_base("SVM")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
    model = LinearSVC()

    model_svc = model.fit(x_train, y_train)
    svc_pred = model_svc.predict(x_test)

    print("Accuracy do SVM Classifier: {}%".format(round(accuracy_score(y_test, svc_pred) * 100, 2)))
    print("\nMatrix de confusão SVM:\n")
    print(confusion_matrix(y_test, svc_pred))
    print("\nDetalhes da classificação SVM :\n")
    print(classification_report(y_test, svc_pred))

def naive_bayesCross():
    model = MultinomialNB()
    k_fold_cross("Naive", model)

def svmCross():
    model = LinearSVC()
    k_fold_cross("SVM", model)

def k_fold_cross(algorithm,model):
    aucs_score =[]
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



def draw_cv_roc_curveGlove():
    x, y = get_base("Glove")
    MAX_NB_WORDS = 100000  # maximo de palavras tokenrizadas
    MAX_SEQUENCE_LENGTH = 1000  # max length of each sentences, including padding
    VALIDATION_SPLIT = 0.2  # 20% of data for validation (not used in training)
    EMBEDDING_DIM = 50  # embedding dimensions for word vectors
    GLOVE_DIR = "C:/Users/gabis/PycharmProjects/Glove/input/glove_s50.txt"

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(x)
    sequences = tokenizer.texts_to_sequences(x)
    word_index = tokenizer.word_index
    print('Vocabulary size:', len(word_index))

    data = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', y.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = y[indices]

    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[: -num_validation_samples]
    y_train = labels[: -num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    print('Number of entries in each category:')
    print('training: ', y_train.sum(axis=0))
    print('validation: ', y_val.sum(axis=0))

    print('Tokenized sentences: \n', data[10])
    print('One hot label: \n', labels[10])

    embeddings_index = {}
    f = open(GLOVE_DIR, encoding='utf8')
    print('Loading Glove from:', GLOVE_DIR, '…', end='')
    for line in f:
        try:
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype='float32')
        except:
            print()
    f.close()
    print("Done.\n Proceeding with Embedding Matrix…", end="")
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("Completed!")

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
                              tf.keras.metrics.FalseNegatives(name='fn')])
    print(tf.math.confusion_matrix(y_val, model.predict(x_val)))

    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))
    result = model.evaluate(x_val, y_val)
    accuracy = result[1]

    print(f"[+] Accuracy: {accuracy * 100:.2f}%")
    print(pd.DataFrame(history.history))
    #print(tf.math.confusion_matrix(y_val, result))
    train_predictions_baseline = model.predict(x_train, batch_size=128)
    test_predictions_baseline = model.predict(x_val, batch_size=128)
    fp, tp, _ = sklearn.metrics.roc_curve(labels, result)



def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))

def naive_bayes_all_validations():
    draw_cv_roc_curve(MultinomialNB())

def svm_all_validations():
    draw_cv_roc_curve(LinearSVC())

def draw_cv_roc_curve(model):

    X, y = get_base("")
    classifier = model

    # #############################################################################
    # Classification and ROC analysis
    # Run classifier with cross-validation and plot ROC curves
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

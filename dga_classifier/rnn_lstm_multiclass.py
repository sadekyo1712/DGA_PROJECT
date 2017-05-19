# Bui Duc Hung - KSCLC HTTT&TT K57 - BKHN - 5/2017
# DGA Classify Project

import dga_classifier.data_generator as data
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
import keras
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import cPickle as pickle
import os

# Show version of framework
print '[*] Keras version: %s' % keras.__version__

class Vectorizer:

    def __init__(self, char_dict):
        self.char_dict = char_dict

    def vectorizer_input(self, input_matrix):
        input_vectorized = [[self.char_dict[character] for character in domain] for domain in input_matrix]
        length = np.max([len(x) for x in input_matrix])
        input_vectorized = sequence.pad_sequences(input_vectorized, maxlen=length)
        return input_vectorized

# Save model to use later
def save_model_to_disk(model_name, model, model_dir='./models'):

    # Serialize model object to save to file
    pickling_model = pickle.dumps(model, pickle.HIGHEST_PROTOCOL)
    # Model path to save model after training
    model_save_path = os.path.join(model_dir, model_name + '.dgaprojectmodel')

    print '[>>>] Storing pickling model to disk %s: %.2f Mb ' % (model_name + '.dgaprojectmodel',
                                                           len(pickling_model) / (1024*1024))
    open(model_save_path, 'wb').write(pickling_model)

def load_model_from_disk(model_name, model_dir='./models'):

    # Path to load model
    model_load_path = os.path.join(model_dir, model_name + '.dgaprojectmodel')
    print '[>>>] Loading model %s from path : %s........' % (model_name, model_load_path)

    # Unpickling model from model_load_path
    try:
        model = pickle.load(open(model_load_path, 'rb').read())
    except:
        print '[XXX] Cannot load model %s from path : %s' % (model_name, model_load_path)
        return None

    return model

def buid_model(input_range, maxlen, number_class=1):

    model = Sequential()
    # Convert discrete space vector to continious space vector to use with LSTM featureless,
    # input is 2D tensor with shape (batch_size, input_length)
    model.add(Embedding(input_dim=input_range, output_dim=128, input_length=maxlen))
    # Don't need to Flatten output 3D tensor (batchsize, input_length, output_dim),
    # since LSTM is a RNN with memory cell
    model.add(LSTM(units=128))
    # I set dropout rate equal 0.5 to prevent overfitting in neural network models
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=number_class))
    model.add(Activation('softmax'))

    # Config optimize and loss of models
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def excute(epoch=25, k_fold=10, batch_size=128):

    data_training = data.get_training_data()

    # Extract data, X with domain, y with labels
    X = [x[0] for x in data_training]
    labels = [x[1] for x in data_training]

    # Build dictionary of characters
    char_dict = {x: index + 1 for index, x in enumerate(set(''.join(X))) }
    # Add 1 to len(char_dict) for special character
    input_range = len(char_dict) + 1
    maxlen = np.max([len(x) for x in X])
    lstm_vectorizer = Vectorizer(char_dict)

    # Convert list domains to vector and padding zeros to get list vector with length = maxlen
    X = [[char_dict[character] for character in domain] for domain in X]
    X = sequence.pad_sequences(X, maxlen=maxlen)

    # For easy mapping
    name_class = ['legit', 'banjori', 'corebot', 'cryptolocker', 'dircrypt', 'kraken', 'locky',
                  'pykspa', 'qakbot', 'ramdo', 'ramnit', 'simda']
    label_dict = {}
    label_index_to_name_dict = {}
    for idx, name in zip(range(len(name_class)), name_class):
        label_dict[name] = idx
        label_index_to_name_dict[idx] = name
    # Indexing labels
    y = [label_dict[name] for name in labels]
    index_labels = y

    # Convert label for multiclass classifier using one hot encoding
    y = keras.utils.to_categorical(y=y, num_classes=len(name_class))

    report_result = []
    last_model = None

    print '\n*********** Multi CLASSIFIER DGA BY LSTM RNN ( FEATURELESS ) ***************'
    print '[*] Preparing....'

    for fold in range(k_fold):
        print '>>>>>> Fold %d/%d' % (fold+1, k_fold)
        X_train, X_test, y_train, y_test, _, labels_test = train_test_split(X, y, index_labels, test_size=0.2)

        print '[*] Build models neural network for training.........'
        model = buid_model(input_range=input_range, maxlen=maxlen, number_class=len(name_class))

        print '[*] Training data.....( using holdout sampling )'
        X_train_holdout, X_test_holdout, y_train_holdout, y_test_holdout = train_test_split(X_train,
                                                                                            y_train,
                                                                                            test_size=0.06)

        best_epoch = -1
        best_auc_score = 0.0
        temp_report = {}

        for ep in range(epoch):

            # Traing data
            model.fit(X_train_holdout, y_train_holdout, batch_size=batch_size, epochs=1)
            # Calculate predict proba and auc score
            probalities_training = model.predict_proba(X_test_holdout)
            pred_class = model.predict_classes(X_test_holdout)

            # Calculate micro AUC
            tpr = {}
            fpr = {}
            roc_auc = {}
            for name in name_class:
                fpr[name], tpr[name], _ = roc_curve(y_test_holdout[:,label_dict[name]],
                                                    probalities_training[:,label_dict[name]])
                roc_auc[name] = auc(fpr[name], tpr[name])
            fpr['micro'], tpr['micro'], _ = roc_curve(y_test_holdout.ravel(), probalities_training.ravel())
            roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

            auc_score = roc_auc['micro']

            print '\n[*] Epoch %d: micro auc score = %f ( best result = %f )' % (ep, auc_score, best_auc_score)
            if auc_score > best_auc_score:
                best_epoch = ep
                best_auc_score = auc_score

                # predict on X_test and cal confusion matrix
                probalities = model.predict_proba(X_test)
                pred_class = model.predict_classes(X_test)
                confusion_matrix = sklearn.metrics.confusion_matrix(labels_test, pred_class, label_dict.values() )

                temp_report = {'y': y_test, 'labels': labels_test, 'probs': probalities, 'epoch': ep,
                               'confusion_matrix': confusion_matrix, 'name_to_index': label_dict,
                               'index_to_name': label_index_to_name_dict}
                print '\n[*] Confusion matrix on epoch %d :' % ep
                print confusion_matrix
            else:
                if (ep - best_epoch) > 2:
                    break

        print '>>>>> End folf %d' % fold
        # Save last model
        if fold == (k_fold - 1):
            last_model = model
        report_result.append(temp_report)

    try:
        print '[*] Save model to hard driver :'
        save_model_to_disk('neural_network_lstm_model', last_model)
        save_model_to_disk('lstm_vectorizer', lstm_vectorizer)
    except Exception:
        print '[XXX] Cannot save model to hard driver :('

    print '[*] Finish training data and buiding neural network models'
    print '-' * 100
    return report_result
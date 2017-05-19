# Bui Duc Hung - KSCLC HTTT&TT K57 - BKHN - 5/2017
# DGA Classify Project

import dga_classifier.data_generator as data
from keras.layers.core import Dense
from keras.models import Sequential
import sklearn
import keras
from sklearn import feature_extraction
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import cPickle as pickle
import os

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

def buid_model(number_feature, number_class=1):

    model = Sequential()
    # neural network to process 2D tensor input, use logistic regression models
    model.add(Dense(number_class, input_dim=number_feature, activation='softmax'))
    # config
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def excute(epoch=70, k_fold=10, batch_size=128):

    data_training = data.get_training_data()

    # Extract data, X with domain, y with labels
    X = [x[0] for x in data_training]
    labels = [x[1] for x in data_training]

    # To use bigram, we need to convert input_data to vector by using ngram
    print '*********** CLASSIFIER Multi DGA BY BIGRAM NEURAL NETWORK ***************'
    print 'Preparing to vectorize input data....'
    print 'Data will be extracted by ngram to extract features and convert to matrix-term-document...'
    # Prepare count_vectorize
    neural_ngram_count_vectorized = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2, 2))
    data_training_vectorized = neural_ngram_count_vectorized.fit_transform(X)
    number_feature = data_training_vectorized.shape[1]

    print '-' * 100
    print ' Analysis data training after vectorized.......'
    print ' Number domain : %d' % data_training_vectorized.shape[0]
    print ' Number dimension : %d' % data_training_vectorized.shape[1]
    print ' List feature : %r' % neural_ngram_count_vectorized.get_feature_names()
    print '-' * 100

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

    # Convert label for multiclass classifier using one-hot-encoding
    y = keras.utils.to_categorical(y=y, num_classes=len(name_class))

    report_result = []
    last_model = None

    for fold in range(k_fold):
        print '>>>>>> Fold %d/%d' % (fold+1, k_fold)
        X_train, X_test, y_train, y_test, _, labels_test = train_test_split(data_training_vectorized,
                                                                            y, index_labels, test_size=0.2)

        print '[*] Build models neural network for training.........'
        model = buid_model(number_feature=number_feature, number_class=len(name_class))

        print '[*] Training data.....( using holdout sampling )'
        X_train_holdout, X_test_holdout, y_train_holdout, y_test_holdout = train_test_split(X_train,
                                                                                            y_train,
                                                                                            test_size=0.05)

        best_epoch = -1
        best_auc_score = 0.0
        temp_report = {}

        for ep in range(epoch):

            # Traing data
            model.fit(X_train_holdout.todense(), y_train_holdout, batch_size=batch_size, epochs=1)
            # Calculate predict proba and auc score
            probalities_training = model.predict_proba(X_test_holdout.todense())
            pred_class = model.predict_classes(X_test_holdout.todense())

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
                probalities = model.predict_proba(X_test.todense())
                pred_class = model.predict_classes(X_test.todense())
                confusion_matrix = sklearn.metrics.confusion_matrix(labels_test, pred_class, label_dict.values() )

                temp_report = {'y': y_test, 'labels': labels_test, 'probs': probalities, 'epoch': ep,
                               'confusion_matrix': confusion_matrix, 'name_to_index': label_dict,
                               'index_to_name': label_index_to_name_dict}
                print '\n[*] Confusion matrix on epoch %d :' % ep
                print confusion_matrix
            else:
                if (ep - best_epoch) > 6:
                    break

        print '>>>>> End folf %d' % fold
        if fold == (k_fold - 1):
            last_model = model
        report_result.append(temp_report)

    try:
        print '[*] Save model to hard driver :'
        save_model_to_disk('neural_network_bigram_multiclass_model', last_model)
        save_model_to_disk('bigram_count_vectorizer_multiclass', neural_ngram_count_vectorized)
    except Exception:
        print '[XXX] Cannot save model to hard driver :('

    print '[*] Finish training data and buiding neural network models.'
    print '-' * 100
    return report_result

# Bui Duc Hung - KSCLC HTTT&TT K57 - BKHN - 5/2017
# DGA Classify Project

import os, sys
import cPickle as pickle
import collections
import sklearn
import sklearn.feature_extraction
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np
import tldextract
import math
import traceback
import matplotlib.pyplot as plt
import dga_classifier.data_generator as data
import operator
from sklearn.preprocessing import label_binarize
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Show verison of framework
print '[*] Scikit learn version: %s' % sklearn.__version__
print '[*] Pandas version: %s' % pd.__version__
print '[*] TLDExtract version: %s' % tldextract.__version__
print '[*] Numpy version: %s' % np.__version__
print '[*] cPickle version: %s' % pickle.__version__

ALEXA_DATA_CSV = './data/alexa_100k'
IMAGE_DATA_ANALYZE_PATH = './image_analysis/'

# Extract second level domain to use
def domain_extract(uri):
    extract_domain = tldextract.extract(uri)
    if (not extract_domain.suffix):
        return None
    else:
        return extract_domain.domain

# Calculate entropy of domain
def entropy(domain):
    # Use Counter to count frequency of each character in domain
    frequency_character_dict, domain_length = collections.Counter(domain), float(len(domain))
    return -sum(count/domain_length * math.log(count/domain_length, 2) for count in frequency_character_dict.values())

def show_confusion_matrix(confusion_matrix, labels):
    # Convert to percent from confusion matrix
    percent_convert = (confusion_matrix * 100) / np.array(np.matrix(confusion_matrix.sum(axis=1)).T)

    print '[*] Confusion matrix status :'
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print '>> Ratio %s/%s is: %.2f%% ( %d/%d )' % (label_j, label_i, percent_convert[i][j],
                                                           confusion_matrix[i][j], confusion_matrix[i].sum())
    print '-' * 30

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

def excute():

    # Excute analyzing data , building feature and building model to classify dga domain by random forest
    final_report = dict()

    try:

        print '*********** CLASSIFIER DGA BY RANDOM FOREST ***************'
        print '[*] Loading pandas dataframe.....'
        data_training = data.get_training_data()

        # Extract data, X with domain, y with labels
        X = [x[0] for x in data_training]
        labels = [x[1] for x in data_training]
        binary_labels = ['legit' if x == 'legit' else 'dga' for x in labels]
        domain_dict = {'domain': X, 'class': labels, 'bin_class': binary_labels}

        # Build pandas DataFrame
        dataframe = pd.DataFrame(domain_dict)
        dataframe = dataframe.dropna()
        dataframe = dataframe.drop_duplicates()
        print '[*] DataFrame generate info :'
        print dataframe.info()

        # Shuffle data for training and testing
        dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
        print '[*] Shuffle dataframe data.....'
        print '[*] Dataframe top 20 domain:'
        print dataframe.head(n=20)

        # Condition for dataframe classify
        condition_legit_domain = dataframe['class'] == 'legit'
        condition_dga_domain = ~condition_legit_domain
        condition_banjori_domain = dataframe['class'] == 'banjori'
        condition_corebot_domain = dataframe['class'] == 'corebot'
        condition_cryptolocker_domain = dataframe['class'] == 'cryptlocker'
        condition_dircrypt_domain = dataframe['class'] == 'dircrypt'
        condition_kraken_domain = dataframe['class'] == 'kraken'
        condition_locky_domain = dataframe['class'] == 'locky'
        condition_pykspa_domain = dataframe['class'] == 'pykspa'
        condition_qakbot_domain = dataframe['class'] == 'qakbot'
        condition_ramdo_domain = dataframe['class'] == 'ramdo'
        condition_ramnit_domain = dataframe['class'] == 'ramnit'
        condition_simda_domain = dataframe['class'] == 'simda'

        print '[*] Total legit ( Alexa based ) domain : %d' % dataframe[condition_legit_domain].shape[0]
        print '[*] Total dga domain : %d' % dataframe[condition_dga_domain].shape[0]

        # Add length field to dataframe
        dataframe['length'] = [len(x) for x in dataframe['domain']]

        # Calculate and add entropy field to dataframe
        dataframe['entropy'] = [entropy(domain=domain) for domain in dataframe['domain']]
        print '[*] Show complete dataframe top 50 :'
        print dataframe.head(n=50)

        # Draw boxplot for domain length group by class
        plt.clf()
        plt.close()
        dataframe.boxplot('length', 'class')
        plt.ylabel('Dataframe domain length')
        dataframe.boxplot('entropy', 'class')
        plt.ylabel('Dataframe domain entropy')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'box_plot_domain_length_entropy_class.png')

        # Draw boxplot for domain length and entropy group by bin_class
        plt.clf()
        plt.close()
        dataframe.boxplot('length', 'bin_class')
        plt.ylabel('Dataframe domain length')
        dataframe.boxplot('entropy', 'bin_class')
        plt.ylabel('Dataframe domain entropy')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'box_plot_domain_length_entropy_bin_class.png')

        # Plot Scatter (length, entropy) for dga and legit
        dga_domain = dataframe[condition_dga_domain]
        legit_domain = dataframe[condition_legit_domain]
        banjori_domain = dataframe[condition_banjori_domain]
        corebot_domain = dataframe[condition_corebot_domain]
        cryptolocker_domain = dataframe[condition_cryptolocker_domain]
        dircrypt_domain = dataframe[condition_dircrypt_domain]
        kraken_domain = dataframe[condition_kraken_domain]
        locky_domain = dataframe[condition_locky_domain]
        pykspa_domain = dataframe[condition_pykspa_domain]
        qakbot_domain = dataframe[condition_qakbot_domain]
        ramdo_domain = dataframe[condition_ramdo_domain]
        ramnit_domain = dataframe[condition_ramnit_domain]
        simda_domain = dataframe[condition_simda_domain]

        plt.clf()
        plt.close()
        plt.scatter(legit_domain['length'], legit_domain['entropy'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(dga_domain['length'], dga_domain['entropy'], s=40, c='#60004a', label='DGA', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain length')
        plt.ylabel('Dataframe domain entropy')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'scatter_entropy_length_binary_class.png')

        plt.clf()
        plt.close()
        plt.scatter(legit_domain['length'], legit_domain['entropy'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(banjori_domain['length'], banjori_domain['entropy'], s=270, c='#003e23', label='banjori', alpha=.3)
        plt.scatter(corebot_domain['length'], corebot_domain['entropy'], s=240, c='#00263e', label='corebot', alpha=.3)
        plt.scatter(cryptolocker_domain['length'], cryptolocker_domain['entropy'], s=210, c='#f4cfeb', label='cryptolocker', alpha=.3)
        plt.scatter(dircrypt_domain['length'], dircrypt_domain['entropy'], s=180, c='#460060', label='dircrypt', alpha=.3)
        plt.scatter(kraken_domain['length'], kraken_domain['entropy'], s=150, c='#968888', label='kraken', alpha=.3)
        plt.scatter(locky_domain['length'], locky_domain['entropy'], s=120, c='#112233', label='locky_v2', alpha=.3)
        plt.scatter(pykspa_domain['length'], pykspa_domain['entropy'], s=90, c='#0e2f44', label='pykspa', alpha=.3)
        plt.scatter(qakbot_domain['length'], qakbot_domain['entropy'], s=60, c='#fb8d8b', label='qakbot', alpha=.3)
        plt.scatter(ramdo_domain['length'], ramdo_domain['entropy'], s=30, c='#033e7b', label='ramdo', alpha=.3)
        plt.scatter(ramnit_domain['length'], ramnit_domain['entropy'], s=15, c='#ffde56', label='ramnit', alpha=.3)
        plt.scatter(simda_domain['length'], simda_domain['entropy'], s=5, c='#f82831', label='simda', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain length')
        plt.ylabel('Dataframe domain entropy')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'scatter_entropy_length_multiclass.png')

        # Build feature ( NGRAM base ),we use ngram for chacracter with n in range 3,4,5
        # and vectorized input domain through feature extract from domain list

        # Build legit base feature
        legit_count_vectorized = sklearn.feature_extraction.text.CountVectorizer(analyzer='char',
                                                                                 ngram_range=(3,5),
                                                                                 min_df=1e-4, max_df=1.0)
        legit_domain_matrix = legit_count_vectorized.fit_transform(legit_domain['domain'])
        count_each_feature_legit_domain_matrix = np.log10(legit_domain_matrix.sum(axis=0).getA1())
        feature_list_legit_domain = legit_count_vectorized.get_feature_names()

        sorted_count_each_feature_legit_domain = sorted(zip(feature_list_legit_domain, count_each_feature_legit_domain_matrix),
                                                        key=operator.itemgetter(1),
                                                        reverse=True)
        print '[*] Legit number ngram feature: %d' % len(sorted_count_each_feature_legit_domain)
        number_feature_show = 20
        print '>> Top %d highest count ngram feature :' % number_feature_show
        print '%15s %10s' % ('ngram-feature', 'count')
        print '-' * 30
        for ngram_feature, count in sorted_count_each_feature_legit_domain[:number_feature_show]:
            print '%15s %10f' % (ngram_feature, count)
        print '-' * 30

        # Build dga base feature
        dga_count_vectorized = sklearn.feature_extraction.text.CountVectorizer( analyzer='char',
                                                                                ngram_range=(3,5),
                                                                                min_df=1e-4, max_df=1.0)
        dga_domain_matrix = dga_count_vectorized.fit_transform(dga_domain['domain'])
        count_each_feature_dga_domain_matrix = np.log10(dga_domain_matrix.sum(axis=0).getA1())
        feature_list_dga_domain = dga_count_vectorized.get_feature_names()

        sorted_count_each_feature_dga_domain = sorted( zip(feature_list_dga_domain, count_each_feature_dga_domain_matrix),
                                                       key=operator.itemgetter(1),
                                                       reverse=True)
        print '[*] DGA number ngram feature: %d' % len(sorted_count_each_feature_dga_domain)
        # number_feature_show = 10
        print '>> Top %d highest count ngram feature :' % number_feature_show
        print '%15s %10s' % ('ngram-feature', 'count')
        print '-' * 30
        for ngram_feature, count in sorted_count_each_feature_dga_domain[:number_feature_show]:
            print '%15s %10f' % (ngram_feature, count)
        print '-' * 30

        # Build dictionary base feature
        # Load dictionary for calculate later
        print '[*] Loading dictionary from ~/data/dictionary.txt .....'
        dictionary_dataframe = pd.read_csv('./data/dictionary.txt', names=['word'], header=None,
                                           dtype={'word': np.str}, encoding='utf-8')
        # Preprocess dictionary_dataframe before use to build training data
        dictionary_dataframe = dictionary_dataframe[dictionary_dataframe['word'].map(lambda x: str(x).isalpha())]
        dictionary_dataframe = dictionary_dataframe.applymap(lambda x: str(x).strip().lower())
        dictionary_dataframe = dictionary_dataframe.dropna()
        dictionary_dataframe = dictionary_dataframe.drop_duplicates()
        print '[*] Dictionary after preprocessing :'
        print dictionary_dataframe.head(n=10)

        # Build count_vectorizer for dictionary to calculate dictionary score for domain
        dictionary_count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(analyzer='char',
                                                                                      ngram_range=(3, 5),
                                                                                      min_df=1e-5,
                                                                                      max_df=1.0)
        dictionary_domain_matrix = dictionary_count_vectorizer.fit_transform(dictionary_dataframe['word'])
        count_each_feature_dictionary_domain_matrix = np.log10(dictionary_domain_matrix.sum(axis=0).getA1())
        feature_list_dictionary_domain = dictionary_count_vectorizer.get_feature_names()

        sorted_count_each_feature_dict_domain = sorted(zip(feature_list_dictionary_domain, count_each_feature_dictionary_domain_matrix),
                                                        key=operator.itemgetter(1),
                                                        reverse=True)
        print '[*] Dictionary number ngram feature: %d' % len(sorted_count_each_feature_dict_domain)
        print '>> Top %d highest count ngram feature :' % number_feature_show
        print '%15s %10s' % ('ngram-feature', 'count')
        print '-' * 30
        for ngram_feature, count in sorted_count_each_feature_dict_domain[:number_feature_show]:
            print '%15s %10f' % (ngram_feature, count)
        print '-' * 30

        def ngram_domain_score(domain):
            legit_score = count_each_feature_legit_domain_matrix * legit_count_vectorized.transform([domain]).T
            dict_score = count_each_feature_dictionary_domain_matrix * dictionary_count_vectorizer.transform([domain]).T
            dga_score = count_each_feature_dga_domain_matrix * dga_count_vectorized.transform([domain]).T
            print '>>>> Domain \'%s\' :\n    $legit_score = %f\n    $dict_score = %f\n    $dga_score = %f' \
                  % (domain, legit_score, dict_score, dga_score)

        # Test domain_score in some popular domains
        ngram_domain_score(domain='google')
        ngram_domain_score(domain='facebook')
        ngram_domain_score(domain='vnexpress')
        ngram_domain_score(domain='tinhte')
        ngram_domain_score(domain='kenh14')
        ngram_domain_score(domain='zing')
        ngram_domain_score(domain='chiasenhac')

        # Calculate domain score for all domain
        dataframe['legit_score'] = count_each_feature_legit_domain_matrix * \
                                   legit_count_vectorized.transform(dataframe['domain']).T
        dataframe['dict_score'] = count_each_feature_dictionary_domain_matrix * \
                                  dictionary_count_vectorizer.transform(dataframe['domain']).T
        dataframe['dga_score'] = count_each_feature_dga_domain_matrix * \
                                 dga_count_vectorized.transform(dataframe['domain']).T


        # Show divegence between legit and dict domain
        # dataframe['legit_score'] > dataframe['dict_score'] => more legit
        # otherwise => more dict
        dataframe['divegence_legit_dict'] = dataframe['legit_score'] - dataframe['dict_score']
        # Show divegence betwwen legit and dga domain
        # dataframe['legit_score'] > dataframe['dga_score'] => more legit
        # otherwise => more dga
        dataframe['divegence_legit_dga'] = dataframe['legit_score'] - dataframe['dga_score']

        # Domain more dictionary than web
        print '[*] Recognize the domains that are more dictionary than web through div_legit_dict'
        print dataframe.sort_values(by=['divegence_legit_dict'], ascending=True, kind='quicksort').head(n=10)
        # Domain more web than dictionary
        print '[*] Recognize the domains that are more web than dictionary through div_legit_dict'
        print dataframe.sort_values(by=['divegence_legit_dict'], ascending=False, kind='quicksort').head(n=10)

        # Domain more dga than legit
        print '[*] Recognize the domains that are more dga than legit through div_legit_dga'
        print dataframe.sort_values(by=['divegence_legit_dga'], ascending=True, kind='quicksort').head(n=10)
        # Domain more legit than dga
        print '[*] Recognize the domains that are more legit than dga through div_legit_dga'
        print dataframe.sort_values(by=['divegence_legit_dga'], ascending=False, kind='quicksort').head(n=10)

        # Visualize effect of divergence
        # Rebuild sub-dataframe
        dga_domain = dataframe[condition_dga_domain]
        legit_domain = dataframe[condition_legit_domain]
        banjori_domain = dataframe[condition_banjori_domain]
        corebot_domain = dataframe[condition_corebot_domain]
        cryptolocker_domain = dataframe[condition_cryptolocker_domain]
        dircrypt_domain = dataframe[condition_dircrypt_domain]
        kraken_domain = dataframe[condition_kraken_domain]
        locky_domain = dataframe[condition_locky_domain]
        pykspa_domain = dataframe[condition_pykspa_domain]
        qakbot_domain = dataframe[condition_qakbot_domain]
        ramdo_domain = dataframe[condition_ramdo_domain]
        ramnit_domain = dataframe[condition_ramnit_domain]
        simda_domain = dataframe[condition_simda_domain]

        # legit_score plot
        plt.clf()
        plt.close()
        plt.scatter(legit_domain['length'], legit_domain['legit_score'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(dga_domain['length'], dga_domain['legit_score'], s=40, c='#60004a', label='DGA', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain length')
        plt.ylabel('Dataframe domain legit_score')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'legit_score_length.png')

        plt.clf()
        plt.close()
        plt.scatter(legit_domain['entropy'], legit_domain['legit_score'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(dga_domain['entropy'], dga_domain['legit_score'], s=40, c='#60004a', label='DGA', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain entropy')
        plt.ylabel('Dataframe domain legit_score')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'legit_score_entropy.png')

        plt.clf()
        plt.close()
        plt.scatter(legit_domain['length'], legit_domain['legit_score'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(banjori_domain['length'], banjori_domain['legit_score'], s=270, c='#003e23', label='banjori', alpha=.3)
        plt.scatter(corebot_domain['length'], corebot_domain['legit_score'], s=240, c='#00263e', label='corebot', alpha=.3)
        plt.scatter(cryptolocker_domain['length'], cryptolocker_domain['legit_score'], s=210, c='#f4cfeb', label='cryptolocker', alpha=.3)
        plt.scatter(dircrypt_domain['length'], dircrypt_domain['legit_score'], s=180, c='#460060', label='dircrypt', alpha=.3)
        plt.scatter(kraken_domain['length'], kraken_domain['legit_score'], s=150, c='#968888', label='kraken', alpha=.3)
        plt.scatter(locky_domain['length'], locky_domain['legit_score'], s=120, c='#112233', label='locky_v2', alpha=.3)
        plt.scatter(pykspa_domain['length'], pykspa_domain['legit_score'], s=90, c='#0e2f44', label='pykspa', alpha=.3)
        plt.scatter(qakbot_domain['length'], qakbot_domain['legit_score'], s=60, c='#fb8d8b', label='qakbot', alpha=.3)
        plt.scatter(ramdo_domain['length'], ramdo_domain['legit_score'], s=30, c='#033e7b', label='ramdo', alpha=.3)
        plt.scatter(ramnit_domain['length'], ramnit_domain['legit_score'], s=15, c='#ffde56', label='ramnit', alpha=.3)
        plt.scatter(simda_domain['length'], simda_domain['legit_score'], s=5, c='#f82831', label='simda', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain length')
        plt.ylabel('Dataframe domain legit_score')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'legit_score_length_multi.png')

        plt.clf()
        plt.close()
        plt.scatter(legit_domain['entropy'], legit_domain['legit_score'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(banjori_domain['entropy'], banjori_domain['legit_score'], s=270, c='#003e23', label='banjori', alpha=.3)
        plt.scatter(corebot_domain['entropy'], corebot_domain['legit_score'], s=240, c='#00263e', label='corebot', alpha=.3)
        plt.scatter(cryptolocker_domain['entropy'], cryptolocker_domain['legit_score'], s=210, c='#f4cfeb', label='cryptolocker', alpha=.3)
        plt.scatter(dircrypt_domain['entropy'], dircrypt_domain['legit_score'], s=180, c='#460060', label='dircrypt', alpha=.3)
        plt.scatter(kraken_domain['entropy'], kraken_domain['legit_score'], s=150, c='#968888', label='kraken', alpha=.3)
        plt.scatter(locky_domain['entropy'], locky_domain['legit_score'], s=120, c='#112233', label='locky_v2', alpha=.3)
        plt.scatter(pykspa_domain['entropy'], pykspa_domain['legit_score'], s=90, c='#0e2f44', label='pykspa', alpha=.3)
        plt.scatter(qakbot_domain['entropy'], qakbot_domain['legit_score'], s=60, c='#fb8d8b', label='qakbot', alpha=.3)
        plt.scatter(ramdo_domain['entropy'], ramdo_domain['legit_score'], s=30, c='#033e7b', label='ramdo', alpha=.3)
        plt.scatter(ramnit_domain['entropy'], ramnit_domain['legit_score'], s=15, c='#ffde56', label='ramnit', alpha=.3)
        plt.scatter(simda_domain['entropy'], simda_domain['legit_score'], s=5, c='#f82831', label='simda', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain entropy')
        plt.ylabel('Dataframe domain legit_score')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'legit_score_entropy_multi.png')

        # dict_score plot
        plt.clf()
        plt.close()
        plt.scatter(legit_domain['length'], legit_domain['dict_score'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(dga_domain['length'], dga_domain['dict_score'], s=40, c='#60004a', label='DGA', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain length')
        plt.ylabel('Dataframe domain dict_score')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'dict_score_length.png')

        plt.clf()
        plt.close()
        plt.scatter(legit_domain['entropy'], legit_domain['dict_score'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(dga_domain['entropy'], dga_domain['dict_score'], s=40, c='#60004a', label='DGA', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain entropy')
        plt.ylabel('Dataframe domain dict_score')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'dict_score_entropy.png')

        plt.clf()
        plt.close()
        plt.scatter(legit_domain['length'], legit_domain['dict_score'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(banjori_domain['length'], banjori_domain['dict_score'], s=270, c='#003e23', label='banjori', alpha=.3)
        plt.scatter(corebot_domain['length'], corebot_domain['dict_score'], s=240, c='#00263e', label='corebot', alpha=.3)
        plt.scatter(cryptolocker_domain['length'], cryptolocker_domain['dict_score'], s=210, c='#f4cfeb', label='cryptolocker', alpha=.3)
        plt.scatter(dircrypt_domain['length'], dircrypt_domain['dict_score'], s=180, c='#460060', label='dircrypt', alpha=.3)
        plt.scatter(kraken_domain['length'], kraken_domain['dict_score'], s=150, c='#968888', label='kraken', alpha=.3)
        plt.scatter(locky_domain['length'], locky_domain['dict_score'], s=120, c='#112233', label='locky_v2', alpha=.3)
        plt.scatter(pykspa_domain['length'], pykspa_domain['dict_score'], s=90, c='#0e2f44', label='pykspa', alpha=.3)
        plt.scatter(qakbot_domain['length'], qakbot_domain['dict_score'], s=60, c='#fb8d8b', label='qakbot', alpha=.3)
        plt.scatter(ramdo_domain['length'], ramdo_domain['dict_score'], s=30, c='#033e7b', label='ramdo', alpha=.3)
        plt.scatter(ramnit_domain['length'], ramnit_domain['dict_score'], s=15, c='#ffde56', label='ramnit', alpha=.3)
        plt.scatter(simda_domain['length'], simda_domain['dict_score'], s=5, c='#f82831', label='simda', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain length')
        plt.ylabel('Dataframe domain dict_score')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'dict_score_length_multi.png')

        plt.clf()
        plt.close()
        plt.scatter(legit_domain['entropy'], legit_domain['dict_score'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(banjori_domain['entropy'], banjori_domain['dict_score'], s=270, c='#003e23', label='banjori', alpha=.3)
        plt.scatter(corebot_domain['entropy'], corebot_domain['dict_score'], s=240, c='#00263e', label='corebot', alpha=.3)
        plt.scatter(cryptolocker_domain['entropy'], cryptolocker_domain['dict_score'], s=210, c='#f4cfeb', label='cryptolocker', alpha=.3)
        plt.scatter(dircrypt_domain['entropy'], dircrypt_domain['dict_score'], s=180, c='#460060', label='dircrypt', alpha=.3)
        plt.scatter(kraken_domain['entropy'], kraken_domain['dict_score'], s=150, c='#968888', label='kraken', alpha=.3)
        plt.scatter(locky_domain['entropy'], locky_domain['dict_score'], s=120, c='#112233', label='locky_v2', alpha=.3)
        plt.scatter(pykspa_domain['entropy'], pykspa_domain['dict_score'], s=90, c='#0e2f44', label='pykspa', alpha=.3)
        plt.scatter(qakbot_domain['entropy'], qakbot_domain['dict_score'], s=60, c='#fb8d8b', label='qakbot', alpha=.3)
        plt.scatter(ramdo_domain['entropy'], ramdo_domain['dict_score'], s=30, c='#033e7b', label='ramdo', alpha=.3)
        plt.scatter(ramnit_domain['entropy'], ramnit_domain['dict_score'], s=15, c='#ffde56', label='ramnit', alpha=.3)
        plt.scatter(simda_domain['entropy'], simda_domain['dict_score'], s=5, c='#f82831', label='simda', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain entropy')
        plt.ylabel('Dataframe domain dict_score')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'dict_score_entropy_multi.png')

         # dga_score plot
        plt.clf()
        plt.close()
        plt.scatter(legit_domain['length'], legit_domain['dga_score'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(dga_domain['length'], dga_domain['dga_score'], s=40, c='#60004a', label='DGA', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain length')
        plt.ylabel('Dataframe domain dga_score')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'dga_score_length.png')

        plt.clf()
        plt.close()
        plt.scatter(legit_domain['entropy'], legit_domain['dga_score'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(dga_domain['entropy'], dga_domain['dga_score'], s=40, c='#60004a', label='DGA', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain entropy')
        plt.ylabel('Dataframe domain dga_score')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'dga_score_entropy.png')

        plt.clf()
        plt.close()
        plt.scatter(legit_domain['length'], legit_domain['dga_score'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(banjori_domain['length'], banjori_domain['dga_score'], s=270, c='#003e23', label='banjori', alpha=.3)
        plt.scatter(corebot_domain['length'], corebot_domain['dga_score'], s=240, c='#00263e', label='corebot', alpha=.3)
        plt.scatter(cryptolocker_domain['length'], cryptolocker_domain['dga_score'], s=210, c='#f4cfeb', label='cryptolocker', alpha=.3)
        plt.scatter(dircrypt_domain['length'], dircrypt_domain['dga_score'], s=180, c='#460060', label='dircrypt', alpha=.3)
        plt.scatter(kraken_domain['length'], kraken_domain['dga_score'], s=150, c='#968888', label='kraken', alpha=.3)
        plt.scatter(locky_domain['length'], locky_domain['dga_score'], s=120, c='#112233', label='locky_v2', alpha=.3)
        plt.scatter(pykspa_domain['length'], pykspa_domain['dga_score'], s=90, c='#0e2f44', label='pykspa', alpha=.3)
        plt.scatter(qakbot_domain['length'], qakbot_domain['dga_score'], s=60, c='#fb8d8b', label='qakbot', alpha=.3)
        plt.scatter(ramdo_domain['length'], ramdo_domain['dga_score'], s=30, c='#033e7b', label='ramdo', alpha=.3)
        plt.scatter(ramnit_domain['length'], ramnit_domain['dga_score'], s=15, c='#ffde56', label='ramnit', alpha=.3)
        plt.scatter(simda_domain['length'], simda_domain['dga_score'], s=5, c='#f82831', label='simda', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain length')
        plt.ylabel('Dataframe domain dga_score')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'dga_score_length_multi.png')

        plt.clf()
        plt.close()
        plt.scatter(legit_domain['entropy'], legit_domain['dga_score'], s=300, c='#3bc293', label='Legit', alpha=.2 )
        plt.scatter(banjori_domain['entropy'], banjori_domain['dga_score'], s=270, c='#003e23', label='banjori', alpha=.3)
        plt.scatter(corebot_domain['entropy'], corebot_domain['dga_score'], s=240, c='#00263e', label='corebot', alpha=.3)
        plt.scatter(cryptolocker_domain['entropy'], cryptolocker_domain['dga_score'], s=210, c='#f4cfeb', label='cryptolocker', alpha=.3)
        plt.scatter(dircrypt_domain['entropy'], dircrypt_domain['dga_score'], s=180, c='#460060', label='dircrypt', alpha=.3)
        plt.scatter(kraken_domain['entropy'], kraken_domain['dga_score'], s=150, c='#968888', label='kraken', alpha=.3)
        plt.scatter(locky_domain['entropy'], locky_domain['dga_score'], s=120, c='#112233', label='locky_v2', alpha=.3)
        plt.scatter(pykspa_domain['entropy'], pykspa_domain['dga_score'], s=90, c='#0e2f44', label='pykspa', alpha=.3)
        plt.scatter(qakbot_domain['entropy'], qakbot_domain['dga_score'], s=60, c='#fb8d8b', label='qakbot', alpha=.3)
        plt.scatter(ramdo_domain['entropy'], ramdo_domain['dga_score'], s=30, c='#033e7b', label='ramdo', alpha=.3)
        plt.scatter(ramnit_domain['entropy'], ramnit_domain['dga_score'], s=15, c='#ffde56', label='ramnit', alpha=.3)
        plt.scatter(simda_domain['entropy'], simda_domain['dga_score'], s=5, c='#f82831', label='simda', alpha=.3)
        plt.legend()
        plt.xlabel('Dataframe domain entropy')
        plt.ylabel('Dataframe domain dga_score')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'dga_score_entropy_multi.png')

        print '[*] Analyze fields in dataframe :'
        print '>>> Legit domain'
        print dataframe[condition_legit_domain].describe()
        print '>>> Dga domain'
        print dataframe[condition_dga_domain].describe()
        print '-' * 30

        # Plot Histogram max_score of legit domain
        max_ngram = np.maximum(legit_domain['legit_score'], legit_domain['dict_score'])
        # plt.hist(max_ngram, bins=150, histtype='stepfilled', color='#003e23')
        plt.clf()
        plt.close()
        plt.hist(max_ngram, bins=150, histtype='step', color='#f82831')
        plt.suptitle('Histogram of the max legit_dict_score for legit domain')
        plt.xlabel('Max (legit,dictionary) score')
        plt.ylabel('Frequency')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'histogram_legit.png')

        # Plot Histogram dga_score
        plt.clf()
        plt.close()
        plt.hist(np.array(dga_domain['dga_score']), bins=150, histtype='step', color='#003300')
        plt.suptitle('Histogram of the dga_score for dga domain')
        plt.xlabel('dga_score')
        plt.ylabel('Frequency')
        plt.grid(True)
        # plt.show()
        plt.savefig(IMAGE_DATA_ANALYZE_PATH + 'histogram_dga.png')

        # Export dataframe to json/csv
        dataframe.to_json('./export/dataframe.json')
        dataframe.to_csv('./export/dataframe.csv')


        # Finish analyze data and preprocessing step, next step I build
        # model RandomForest to training and classify dga domain

        randomforest_labels = ['legit', 'dga']
        randomforest_multilabels = ['legit', 'banjori', 'corebot', 'cryptolocker', 'dircrypt',
                                    'kraken', 'locky', 'pykspa', 'qakbot', 'ramdo', 'ramnit', 'simda']

        # X -> dataset to training, y -> labels
        X = dataframe.as_matrix(['length', 'entropy', 'legit_score', 'dict_score', 'dga_score'])
        y = np.array(dataframe['bin_class'])
        y_multiclass = np.array(dataframe['class'])

        y_multiclass_binarize = label_binarize(y_multiclass, classes=randomforest_multilabels)
        randomforest_multilabels_binarize = label_binarize(randomforest_multilabels, classes=randomforest_multilabels)

        # RandomForest model to classify
        randomforest_clf = RandomForestClassifier(n_estimators=20, criterion='entropy', bootstrap=True, n_jobs=-1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print '[*] Training data......'
        randomforest_clf.fit(X_train, y_train)

        # For multi class, convert y_test to bin
        y_test_bin = label_binarize(y_test, randomforest_clf.classes_)
        num_class = y_test_bin.shape[1]
        y_test_bin = np.array([x.astype(int) for x in y_test_bin])
        # For transform index <-> class easier
        index_class_dict = {}
        class_index_dict = {}
        for idx, x in zip(range(len(randomforest_clf.classes_)), randomforest_clf.classes_):
            index_class_dict[idx] = x
            class_index_dict[x] = idx

        print '[*] Testing data.....'
        y_predict = randomforest_clf.predict(X_test)
        y_predict_prob = randomforest_clf.predict_proba(X_test)

        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_predict, randomforest_labels)

        print '[*] Show confusion matrix :'
        print confusion_matrix
        print '[*] Analyzing result.......'

        show_confusion_matrix(confusion_matrix, randomforest_labels)

        importances_feature = zip(['length', 'entropy', 'legit_score', 'dict_score', 'dga_score'],
                                  randomforest_clf.feature_importances_)
        print '==>> Importance feature :'
        for feature, importance in importances_feature:
            print '     > Feature \'%s\'(entropy) = %f' % (feature, importance)
        print '[*] List class recognize by randomforest classifier :'
        print randomforest_clf.classes_
        prob_perclass_dict = dict()
        for index, class_name in enumerate(randomforest_clf.classes_):
            prob_perclass_dict[class_name] = y_predict_prob[:,index]

        # Convert label to prob value
        y_test_true_table = [0 if label == 'legit' else 1 for label in y_test]
        y_test_true_table_multi = [1 if label == 'legit' else 0 for label in y_test]

        # Calculate fpr, tpr to plot roc curve and calculate auc
        fpr, tpr, threadshold = sklearn.metrics.roc_curve(y_test_true_table, np.array(prob_perclass_dict['dga']))

        auc = sklearn.metrics.auc(fpr, tpr)
        print '[*] AUC is %f' % auc

        # Train on whole dataframe
        print '[*] Training whole dataset.......'
        randomforest_clf.fit(X, y)
        # randomforest_clf.fit(X, y_multiclass)

        def quick_test(uri):

            domain = domain_extract(uri=uri)
            legit_score = count_each_feature_legit_domain_matrix * legit_count_vectorized.transform([domain]).T
            dict_score = count_each_feature_dictionary_domain_matrix * dictionary_count_vectorizer.transform([domain]).T
            dga_score = count_each_feature_dga_domain_matrix * dga_count_vectorized.transform([domain]).T
            vectorized_domain = [len(domain), entropy(domain), legit_score, dict_score, dga_score]
            print '>>>> Test domain \'%s\' : %s' % (uri, randomforest_clf.predict(vectorized_domain)[0])

        # Test on some domain legit and dga
        print '[*] Test on some domain : '
        quick_test('google.com.vn')
        quick_test('vnexpress.net')
        quick_test('tinhte.vn')
        quick_test('kenh14.vn')
        quick_test('40a43e61e56a5c218cf6c22aca27f7ee.org')
        quick_test('agabgtdhgsbspwsq.ru')
        quick_test('dantri.net')
        quick_test('axtopsbtntqnfdyk.ru')
        quick_test('ahamove.com.vn')
        quick_test('batqeodiji.com')
        quick_test('bdjhtgqhggicwrmy.ru')
        quick_test('melhlehkvxoxbqq.net')
        # WannaCry kill switch domain
        quick_test('www.iuqerfsodp9ifjaposdfjhgosurijfaewrwergwea.com')
        # Same weird domain =)) but i want to compare to WannaCry kill switch domain
        quick_test('www.hungdetraicogisaikhongthebaolakhongdeptrai.com')
        print '==>> Finish testing'

        # Save whole model
        print '[*] Save model to hard driver:'
        save_model_to_disk('legit_count_matrix', count_each_feature_legit_domain_matrix)
        save_model_to_disk('legit_count_vectorizer', legit_count_vectorized)
        save_model_to_disk('dictionary_count_matrix', count_each_feature_dictionary_domain_matrix)
        save_model_to_disk('dictionary_count_vectorizer', dictionary_count_vectorizer)
        save_model_to_disk('dga_count_matrix', count_each_feature_dga_domain_matrix)
        save_model_to_disk('dga_count_vectorizer', dga_count_vectorized)
        save_model_to_disk('random_forest_classifier', randomforest_clf)

        plt.close()
        final_report = {'y': y_test_true_table, 'labels': y_test, 'probs': prob_perclass_dict, 'epoch': 0,
                        'confusion_matrix': confusion_matrix}
    except KeyboardInterrupt:
        print '>>>>>>>> Terminating.....'
        sys.exit(0)
    except Exception, error:
        print '>>>>>>>> Cannot build model :(....'
        print traceback.print_exc()
        print 'Error occur : %s' % (str(error))
        sys.exit(1)

    return final_report



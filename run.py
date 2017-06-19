# Bui Duc Hung - KSCLC HTTT&TT K57 - BKHN - 5/2017
# DGA Classify Project

import os
import cPickle as pickle
from matplotlib import pyplot as plt
import numpy as np
import dga_classifier.neural_ngram as bigram
import dga_classifier.rnn_lstm as lstm
import dga_classifier.random_forest_feature as rf
import dga_classifier.neural_ngram_multiclass as bigram_multi
import dga_classifier.rnn_lstm_multiclass as lstm_multi
import dga_classifier.random_forest_feature_multiclass as rf_multi
from scipy import interp
import itertools
from sklearn.metrics import roc_curve, auc

RESULT = './export/result.pkl'

def evaluate(excute_bigram=True, excute_lstm=True, excute_rf=True,
             excute_bigram_multi=True, excute_lstm_multi=True,
             excute_rf_multi=True, k_fold=10, epoch=70):

    bigram_result = None
    lstm_result = None
    rf_result = None
    bigram_multi_result = None
    lstm_multi_result = None
    rf_multi_result = None


    if epoch < 10:
        # Quick scan to demo and test
        if excute_bigram:
            bigram_result = bigram.excute(k_fold=k_fold, epoch=epoch)
        if excute_lstm:
            lstm_result = lstm.excute(k_fold=k_fold, epoch=epoch)
        if excute_rf:
            rf_result = rf.excute()
        if excute_bigram_multi:
            bigram_multi_result = bigram_multi.excute(k_fold=k_fold, epoch=epoch)
        if excute_lstm_multi:
            lstm_multi_result = lstm_multi.excute(k_fold=k_fold, epoch=epoch)
        if excute_rf_multi:
            rf_multi_result = rf_multi.excute()
    else:
        # Full scan for deploy
        if excute_bigram:
            bigram_result = bigram.excute(k_fold=k_fold)
        if excute_lstm:
            lstm_result = lstm.excute(k_fold=k_fold)
        if excute_rf:
            rf_result = rf.excute()
        if excute_bigram_multi:
            bigram_multi_result = bigram_multi.excute(k_fold=k_fold)
        if excute_lstm_multi:
            lstm_multi_result = lstm_multi.excute(k_fold=k_fold)
        if excute_rf_multi:
            rf_multi_result = rf_multi.excute()

    return rf_result, bigram_result, lstm_result, rf_multi_result, bigram_multi_result, lstm_multi_result

def calculate_macro_roc(fpr, tpr):

    all_fpr = np.unique(np.concatenate(fpr))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(tpr)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(tpr)

    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

def plot_confusion_matrix(confusion_matrix, labels, title='Confusion Matrix'):

    percent_convert = (confusion_matrix * 100) / np.array(np.matrix(confusion_matrix.sum(axis=1)).T)

    print '[*] Confusion matrix of ' + title +' status :'
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print '>> Ratio %s/%s is: %.2f%% ( %d/%d )' % (label_j, label_i, percent_convert[i][j],
                                                           confusion_matrix[i][j], confusion_matrix[i].sum())
    print '-' * 30

    plt.clf()
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    cax = ax.matshow(percent_convert, cmap='coolwarm')
    plt.title(title)
    fig.colorbar(cax)
    tickmark = np.arange(len(labels))
    plt.xticks(tickmark, labels)
    plt.yticks(tickmark, labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    # plt.savefig('./image_analysis/' + title + '.png')


def run(excute_bigram=True, excute_lstm=True, excute_rf=True, excute_bigram_multi=True, excute_lstm_multi=True,
             excute_rf_multi=True, k_fold=10, gate=False, epoch=70):

    print '[*] Loading......'
    if gate or (not os.path.isfile(RESULT)):

        rf_result, bigram_result, lstm_result, \
        rf_multi_result, bigram_multi_result, lstm_multi_result = \
            evaluate(excute_bigram, excute_lstm, excute_rf,
                     excute_bigram_multi, excute_lstm_multi, excute_rf_multi,
                     k_fold=k_fold, epoch=epoch)

        result = {'random_forest': rf_result, 'bigram': bigram_result, 'lstm': lstm_result,
                  'random_forest_multi': rf_multi_result, 'bigram_multi': bigram_multi_result,
                  'lstm_multi': lstm_multi_result}
        pickle.dump(result, open(RESULT, 'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        result = pickle.load(open(RESULT, 'rb'))

    print '[*] Finish evaluate'
    print '[*] Analyzing and building result data.......'
    number_class = None
    index_to_name_rf_dict = None
    index_to_name_bigram_dict = None
    index_to_name_lstm_dict = None
    init_val = []
    init_val += [None] * 9 * 2
    rf_fpr, rf_tpr, rf_auc, bigram_fpr, bigram_tpr, bigram_auc, lstm_fpr, lstm_tpr, lstm_auc,\
    rf_multi_fpr, rf_multi_tpr, rf_multi_auc, bigram_multi_fpr, bigram_multi_tpr, bigram_multi_auc, \
    lstm_multi_fpr, lstm_multi_tpr, lstm_multi_auc = init_val

    if result['random_forest']:
        rf_result = result['random_forest']
        # ________________
        confusion_matrix = rf_result['confusion_matrix']
        labels = ['legit', 'dga']
        plot_confusion_matrix(confusion_matrix, labels, title='Random forest CM')
        # ________________
        y_true_table = rf_result['y']
        prob_dict = rf_result['probs']
        rf_fpr, rf_tpr, _ = roc_curve(y_true_table, prob_dict['dga'])
        rf_auc = auc(rf_fpr, rf_tpr)

    if result['bigram']:
        bigram_result = result['bigram']
        fpr = []
        tpr = []
        for each_bigram_result in bigram_result:
            temp_fpr, temp_tpr, _ = roc_curve(each_bigram_result['y'], each_bigram_result['probs'])
            fpr.append(temp_fpr)
            tpr.append(temp_tpr)
            # ________________
            confusion_matrix = each_bigram_result['confusion_matrix']
            labels = ['legit', 'dga']
            plot_confusion_matrix(confusion_matrix, labels, title='Bigram CM')
            # ________________

        bigram_fpr, bigram_tpr, bigram_auc = calculate_macro_roc(fpr, tpr)

    if result['lstm']:
        lstm_result = result['lstm']
        fpr = []
        tpr = []
        for each_lstm_result in lstm_result:
            temp_fpr, temp_tpr, _ = roc_curve(each_lstm_result['y'], each_lstm_result['probs'])
            fpr.append(temp_fpr)
            tpr.append(temp_tpr)
            # ________________
            confusion_matrix = each_lstm_result['confusion_matrix']
            labels = ['legit', 'dga']
            plot_confusion_matrix(confusion_matrix, labels, title='LSTM CM')
            # ________________

        lstm_fpr, lstm_tpr, lstm_auc = calculate_macro_roc(fpr, tpr)

    if result['random_forest_multi']:
        rf_multi_result = result['random_forest_multi']
        fpr = {}
        tpr = {}
        auc_roc = {}
        y_test_bin = rf_multi_result['y']
        y_predict_prob = rf_multi_result['y_predict_prob']
        y_test = rf_multi_result['labels']
        prob_per_class_dict = rf_multi_result['probs']
        name_to_index_dict = rf_multi_result['name_to_index']
        index_to_name_dict = rf_multi_result['index_to_name']
        # ________________
        confusion_matrix = rf_multi_result['confusion_matrix']
        labels = index_to_name_dict.values()
        plot_confusion_matrix(confusion_matrix, labels, title='Random forest Multi CM')
        # ________________
        number_class = len(name_to_index_dict)

        # tpr, fpr, auc for all class
        for i in range(number_class):
            fpr[index_to_name_dict[i]], tpr[index_to_name_dict[i]], _ = \
                roc_curve(y_test_bin[:,i], prob_per_class_dict[index_to_name_dict[i]])
            auc_roc[index_to_name_dict[i]] = auc(fpr[index_to_name_dict[i]], tpr[index_to_name_dict[i]])

        # Micro auc
        fpr['micro'], tpr['micro'], _ = roc_curve(y_test_bin.ravel(), y_predict_prob.ravel())
        auc_roc['micro'] = auc(fpr['micro'], tpr['micro'])

        # Macro auc
        fpr_list = [fpr[index_to_name_dict[i]] for i in range(number_class)]
        tpr_list = [tpr[index_to_name_dict[i]] for i in range(number_class)]
        fpr['macro'], tpr['macro'], auc_roc['macro'] = calculate_macro_roc(fpr_list, tpr_list)

        rf_multi_fpr = fpr
        rf_multi_tpr = tpr
        rf_multi_auc = auc_roc
        index_to_name_rf_dict = index_to_name_dict

    if result['bigram_multi']:
        bigram_multi_result_list = result['bigram_multi']
        bigram_multi_result = bigram_multi_result_list[0]
        fpr = {}
        tpr = {}
        auc_roc = {}

        y_test = bigram_multi_result['y']
        probalities = bigram_multi_result['probs']
        name_to_index_dict = bigram_multi_result['name_to_index']
        index_to_name_dict = bigram_multi_result['index_to_name']
        # ________________
        confusion_matrix = bigram_multi_result['confusion_matrix']
        labels = index_to_name_dict.values()
        plot_confusion_matrix(confusion_matrix, labels, title='Bigram multi CM')
        # ________________
        number_class = len(name_to_index_dict)
        for i in range(number_class):
            fpr[index_to_name_dict[i]], tpr[index_to_name_dict[i]], _ = roc_curve(y_test[:,i], probalities[:,i])
            auc_roc[index_to_name_dict[i]] = auc(fpr[index_to_name_dict[i]], tpr[index_to_name_dict[i]])

        fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), probalities.ravel())
        auc_roc['micro'] = auc(fpr['micro'], tpr['micro'])

        fpr_list = [fpr[index_to_name_dict[i]] for i in range(number_class)]
        tpr_list = [tpr[index_to_name_dict[i]] for i in range(number_class)]
        fpr['macro'], tpr['macro'], auc_roc['macro'] = calculate_macro_roc(fpr_list, tpr_list)

        bigram_multi_fpr = fpr
        bigram_multi_tpr = tpr
        bigram_multi_auc = auc_roc
        index_to_name_bigram_dict = index_to_name_dict

    if result['lstm_multi']:
        lstm_multi_result_list = result['lstm_multi']
        lstm_multi_result = lstm_multi_result_list[0]
        fpr = {}
        tpr = {}
        auc_roc = {}
        y_test = lstm_multi_result['y']
        probalities = lstm_multi_result['probs']
        name_to_index_dict = lstm_multi_result['name_to_index']
        index_to_name_dict = lstm_multi_result['index_to_name']
        # ________________
        confusion_matrix = lstm_multi_result['confusion_matrix']
        labels = index_to_name_dict.values()
        plot_confusion_matrix(confusion_matrix, labels, title='LSTM multi CM')
        # ________________
        number_class = len(name_to_index_dict)
        for i in range(number_class):
            fpr[index_to_name_dict[i]], tpr[index_to_name_dict[i]], _ = roc_curve(y_test[:,i], probalities[:,i])
            auc_roc[index_to_name_dict[i]] = auc(fpr[index_to_name_dict[i]], tpr[index_to_name_dict[i]])

        fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), probalities.ravel())
        auc_roc['micro'] = auc(fpr['micro'], tpr['micro'])

        fpr_list = [fpr[index_to_name_dict[i]] for i in range(number_class)]
        tpr_list = [tpr[index_to_name_dict[i]] for i in range(number_class)]
        fpr['macro'], tpr['macro'], auc_roc['macro'] = calculate_macro_roc(fpr_list, tpr_list)

        lstm_multi_fpr = fpr
        lstm_multi_tpr = tpr
        lstm_multi_auc = auc_roc
        index_to_name_lstm_dict = index_to_name_dict

    # Param for pyplot
    lw = 2

    # Plot result
    plt.clf()
    plt.close()
    plt.plot(rf_fpr, rf_tpr, label='Randomforest ( AUC = %.4f )' % rf_auc, color='#095115')
    plt.plot(bigram_fpr, bigram_tpr, label='Bigram ( AUC = %.4f )' % bigram_auc, color='#c2473e')
    plt.plot(lstm_fpr, lstm_tpr, label='LSTM ( AUC = %.4f )' % lstm_auc, color='#275c8d')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC random forest / bigram / lstm')
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tick_params(axis='both')
    # plt.show()
    plt.savefig('./image_analysis/results_bin_classifier.png')

    # RF multi
    plt.clf()
    plt.close()
    plt.plot(rf_multi_fpr['micro'], rf_multi_tpr['micro'], label='micro ROC ( AUC = %.4f )'% rf_multi_auc['micro'],
             color='#095115', linestyle=':', linewidth=4)
    plt.plot(rf_multi_fpr['macro'], rf_multi_tpr['macro'], label='macro ROC ( AUC = %.4f )'% rf_multi_auc['macro'],
             color='#c2473e', linestyle=':', linewidth=4)
    colors = itertools.cycle(['#052131', '#fef65b', '#7bcd8a', '#cfa5df', '#2d74b2', '#ab3456',
                              '#fd1f58', '#1ffdc4', '#6a008e', '#433159', '#d46977', '#35c0b2'])
    for i, color in zip(range(number_class), colors):
        plt.plot(rf_multi_fpr[index_to_name_rf_dict[i]], rf_multi_tpr[index_to_name_rf_dict[i]], color=color, lw=lw,
                 label='ROC of class %s - auc = %.4f' % (index_to_name_rf_dict[i],
                                                         rf_multi_auc[index_to_name_rf_dict[i]]) )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Random forest multiclass ROC')
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tick_params(axis='both')
    # plt.show()
    plt.savefig('./image_analysis/results_rf_multi.png')

    # Bigram multi
    plt.clf()
    plt.close()
    plt.plot(bigram_multi_fpr['micro'], bigram_multi_tpr['micro'],
             label='micro ROC ( AUC = %.4f )'% bigram_multi_auc['micro'],
             color='#095115', linestyle=':', linewidth=4)
    plt.plot(bigram_multi_fpr['macro'], bigram_multi_tpr['macro'],
             label='macro ROC ( AUC = %.4f )'% bigram_multi_auc['macro'],
             color='#c2473e', linestyle=':', linewidth=4)
    colors = itertools.cycle(['#052131', '#fef65b', '#7bcd8a', '#cfa5df', '#2d74b2', '#ab3456',
                              '#fd1f58', '#1ffdc4', '#6a008e', '#433159', '#d46977', '#35c0b2'])
    for i, color in zip(range(number_class), colors):
        plt.plot(bigram_multi_fpr[index_to_name_bigram_dict[i]], bigram_multi_tpr[index_to_name_bigram_dict[i]],
                 color=color, lw=lw,
                 label='ROC of class %s - auc = %.4f' % (index_to_name_bigram_dict[i],
                                                         bigram_multi_auc[index_to_name_bigram_dict[i]]) )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Bigram multiclass ROC')
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tick_params(axis='both')
    # plt.show()
    plt.savefig('./image_analysis/results_bigram_multi.png')

    # LSTM multi
    plt.clf()
    plt.close()
    plt.plot(lstm_multi_fpr['micro'], lstm_multi_tpr['micro'],
             label='micro ROC ( AUC = %.4f )'% lstm_multi_auc['micro'],
             color='#095115', linestyle=':', linewidth=4)
    plt.plot(lstm_multi_fpr['macro'], lstm_multi_tpr['macro'],
             label='macro ROC ( AUC = %.4f )'% lstm_multi_auc['macro'],
             color='#c2473e', linestyle=':', linewidth=4)
    colors = itertools.cycle(['#052131', '#fef65b', '#7bcd8a', '#cfa5df', '#2d74b2', '#ab3456',
                              '#fd1f58', '#1ffdc4', '#6a008e', '#433159', '#d46977', '#35c0b2'])
    for i, color in zip(range(number_class), colors):
        plt.plot(lstm_multi_fpr[index_to_name_lstm_dict[i]], lstm_multi_tpr[index_to_name_lstm_dict[i]],
                 color=color, lw=lw,
                 label='ROC of class %s - auc = %.4f' % (index_to_name_lstm_dict[i],
                                                         lstm_multi_auc[index_to_name_lstm_dict[i]]) )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('LSTM multiclass ROC')
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tick_params(axis='both')
    # plt.show()
    plt.savefig('./image_analysis/results_lstm_multi.png')


if __name__ == '__main__':

    # Full scan
    # run(k_fold=1)

    # Quick scan
    run(k_fold=1, epoch=1)

    plt.clf()
    plt.close()



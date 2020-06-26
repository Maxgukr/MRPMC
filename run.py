# -*- coding: utf-8 -*-

from data_preprocess import generate_data, generate_train_data
import pandas as pd
from Model import RF, GBDT, LR, KNN, SVM, StackModel, VotingModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_curve, \
    f1_score, roc_curve, average_precision_score, auc, confusion_matrix, precision_score, recall_score
import datetime as dt
import os
from statsmodels.stats.proportion import proportion_confint
from sklearn.calibration import calibration_curve

import shap_interpretation


X_train, y_train, X_test_zf, y_test_zf, id_zf = generate_train_data('./data/zfprocessed_data.csv',
                                                                    delete_grade=True,
                                                                    over_sample=False)
test_data_gg, test_label_gg, columns, id_gg = generate_data('./data/ggprocessed_data.csv', delete_grade=True)
test_data_xy, test_label_xy, _, id_xy = generate_data('./data/xyprocessed_data.csv', delete_grade=True)


mode = 'stack'
path = './results/'+dt.datetime.now().strftime('%Y%m%d-%H-%M') + '-'+str(len(columns)) + '-' + mode
os.makedirs(path, exist_ok=True)


def gbdt(X_train, y_train, X_test, y_test):
    # GBDT
    gbdt = GBDT(X_train, y_train, X_test, y_test)

    gbdt_feature_importance, estimator_score = gbdt.gbdt_train()

    gbdt_predict_result = gbdt.gbdt_predict()
    gbdt_predict_result_prob = gbdt.gbdt_predict_prob()

    gbdt_score = gbdt.gbdt_score()

    # save rf feature importance
    df_gbdt_feature_importance = pd.DataFrame(data=gbdt_feature_importance.reshape(1, len(gbdt_feature_importance)),
                                              columns=list(columns))
    df_gbdt_feature_importance.sort_values(by=0, axis=1, ascending=False, inplace=True)
    # df_rf_feature_importance.loc[0, df_rf_feature_importance.columns.values[:30]].plot(kind='bar')
    # path = './feature_select/' + dt.datetime.now().strftime('%Y%m%d-%H-%M')
    df_gbdt_feature_importance.to_excel(path+'/'+'gbdt_feature_importance.xlsx', index=False)

    print("gradient boost get score:", gbdt_score)

    return [gbdt, gbdt_predict_result, gbdt_predict_result_prob]


def rf(X_train, y_train, X_test, y_test):
    # RF
    rf = RF(X_train, y_train, X_test, y_test)

    rf_feature_importance, oob_score = rf.rf_train()

    rf_predict_result = rf.rf_predict()
    rf_predict_result_prob = rf.rf_predict_prob()

    rf_score = rf.rf_score()

    # save rf feature importance
    df_rf_feature_importance = pd.DataFrame(data=rf_feature_importance.reshape(1, len(rf_feature_importance)),
                                            columns=list(columns))
    df_rf_feature_importance.sort_values(by=0, axis=1, ascending=False, inplace=True)
    # df_rf_feature_importance.loc[0, df_rf_feature_importance.columns.values[:30]].plot(kind='bar')
    # path = './feature_select/' + dt.datetime.now().strftime('%Y%m%d-%H-%M')
    df_rf_feature_importance.to_excel(path+'/'+'rf_feature_importance.xlsx', index=False)
    print("random forest get score:", rf_score)

    return [rf, rf_predict_result, rf_predict_result_prob]


def lr(X_train, y_train, X_test, y_test):
    # lr
    lr = LR(X_train, y_train, X_test, y_test)
    lr_coef_ = lr.lr_train()
    lr_predict = lr.lr_predict()
    lr_predict_proba = lr.lr_predict_proba()

    # save rf feature importance
    df_lr_feature_importance = pd.DataFrame(data=np.abs(lr_coef_),
                                            columns=list(columns))
    df_lr_feature_importance.sort_values(by=0, axis=1, ascending=False, inplace=True)
    # df_lr_feature_importance.loc[0, df_lr_feature_importance.columns.values].plot(kind='bar')
    # path = './feature_select/' + dt.datetime.now().strftime('%Y%m%d-%H-%M')
    df_lr_feature_importance.to_excel(path+'/'+'lr_feature_importance.xlsx', index=False)
    print("lr get score:", lr.lr_score())
    return [lr, lr_predict, lr_predict_proba]


def knn(X_train, y_train, X_test, y_test):
    knn = KNN(X_train, y_train, X_test, y_test)
    knn.knn_train()
    knn_predict = knn.knn_predict()
    knn_predict_proba = knn.knn_predict_proba()
    print("knn get score:", knn.score())
    return [knn, knn_predict, knn_predict_proba]


def svm(X_train, y_train, X_test, y_test):
    svm = SVM(X_train, y_train, X_test, y_test)
    svm_coef_ = svm.svm_train()
    svm_predict = svm.svm_predict()
    svm_confidence = svm.svm_predict_confidence()
    df_svm_feature_importance = pd.DataFrame(data=np.abs(svm_coef_),
                                             columns=list(columns))
    df_svm_feature_importance.sort_values(by=0, axis=1, ascending=False, inplace=True)
    # df_svm_feature_importance.loc[0, df_svm_feature_importance.columns.values].plot(kind='bar')
    # path = './feature_select/' + dt.datetime.now().strftime('%Y%m%d-%H-%M')
    df_svm_feature_importance.to_excel(path+'/'+'svm_feature_importance.xlsx', index=False)

    print("svm get score:", svm.svm_score())
    return [svm, svm_predict, svm_confidence]


def stack_models(X_train, y_train, X_test, y_test):
    stack_model = StackModel()
    stack_model.train(X_train, y_train)
    stack_model_predict, stack_model_predict_proba = stack_model.predict_proba(X_test, y_test)
    # print("stack get score:", stack_model.score(X_test, y_test))
    return [stack_model, stack_model_predict, stack_model_predict_proba]


def vote_models(X_train, y_train, X_test, y_test):
    vote_model = VotingModel()
    vote_model.train(X_train, y_train)
    vote_model_predict = vote_model.predict(X_test)
    vote_model_predict_proba = vote_model.predict_proba(X_test)
    return [vote_model, vote_model_predict, vote_model_predict_proba]


def voting(rf, gbdt, lr, svm, weights):  # svm,
    voting_proba = np.zeros((len(rf), 2))
    voting_label = np.zeros((len(rf), ))
    all_proba = np.hstack((rf[:, 1].reshape(len(rf), 1),
                           gbdt[:, 1].reshape(len(rf), 1),
                           lr[:, 1].reshape(len(rf), 1),
                           svm[:, 1].reshape(len(svm), 1)))

    for i in range(len(all_proba)):
        voting_proba[i][1] = (all_proba[i][0]*weights[0] + all_proba[i][1]*weights[1] +
                              all_proba[i][2]*weights[2] + all_proba[i][3]*weights[3])/np.sum(weights)  #
        voting_proba[i][0] = 1 - voting_proba[i][1]
        if voting_proba[i][1] >= 0.5:
            voting_label[i] = 2
        else:
            voting_label[i] = 1

    return [rf, voting_label, voting_proba]


summary1 = pd.DataFrame(data=np.zeros((19, 12)),
                        columns=pd.MultiIndex.from_product([['gg', 'zf', 'xy'],
                                                           ['EmsembleModel', 'RF', 'GBDT', 'LR']]),
                        index=['AUC',
                               'AUC-95%-CI-low',
                               'AUC-95%-CI-up',
                               'Accuracy',
                               'Accuracy-95%-CI-low',
                               'Accuracy-95%-CI-up',
                               'Sensitivity',
                               'Sensitivity-95%-CI-low',
                               'Sensitivity-95%-CI-up',
                               'Specificity',
                               'Specificity-95%-CI-low',
                               'Specificity-95%-CI-up',
                               'PPV',
                               'NPV',
                               'f1 score',
                               'TP',
                               'FN',
                               'FP',
                               'TN'])


summary2 = pd.DataFrame(data=np.zeros((19, 6)),
                        columns=pd.MultiIndex.from_product([['gg', 'zf', 'xy'],
                                                           ['SVM', 'KNN']]),
                        index=['AUC',
                               'AUC-95%-CI-low',
                               'AUC-95%-CI-up',
                               'Accuracy',
                               'Accuracy-95%-CI-low',
                               'Accuracy-95%-CI-up',
                               'Sensitivity',
                               'Sensitivity-95%-CI-low',
                               'Sensitivity-95%-CI-up',
                               'Specificity',
                               'Specificity-95%-CI-low',
                               'Specificity-95%-CI-up',
                               'PPV',
                               'NPV',
                               'f1 score',
                               'TP',
                               'FN',
                               'FP',
                               'TN'])


def analysis_results(results, X_test, y_test, hp):
    # file = open(path+'/'+hp+'-summary.txt', 'w')
    # old = sys.stdout
    # sys.stdout = file
    file = str(len(results))
    new_path = path+'/'+file+'/'
    os.makedirs(new_path, exist_ok=True)
    if len(results) == 4:
        summary = summary1
    else:
        summary = summary2

    shap_interpretation.run_shap(results, X_test, new_path, hp)

    '''
    for key, item in results.items():
        print("####################################################")
        print("/*results of ", key, "*/")
        print("confusion_matrix:")
        m = confusion_matrix(y_test - 1, results[key][1] - 1, labels=[1, 0])
        print(m)
        summary.loc['TP'][hp, key] = m[0][0]
        summary.loc['FN'][hp, key] = m[0][1]
        summary.loc['FP'][hp, key] = m[1][0]
        summary.loc['TN'][hp, key] = m[1][1]
        # print("accuracy:", accuracy_score(y_test-1, results[key][1]-1))
        summary.loc['Accuracy'][hp, key] = accuracy_score(y_test-1, results[key][1]-1)
        lower, upper = proportion_confint(m[0][0]+m[1][1], m[0][0]+m[0][1]+m[1][0]+m[1][1], 0.05, method='normal')
        # print("accuracy 95% confidence interval:", lower, upper)
        summary.loc['Accuracy-95%-CI-low'][hp, key] = lower
        summary.loc['Accuracy-95%-CI-up'][hp, key] = upper
        # print("f1 score:", f1_score(y_test-1, results[key][1]-1))
        summary.loc['f1 score'][hp, key] = f1_score(y_test-1, results[key][1]-1)
        # print("PPV:", precision_score(y_test-1, results[key][1]-1))
        summary.loc['PPV'][hp, key] = precision_score(y_test-1, results[key][1]-1)
        # print("NPV:", float(m[1][1]/(m[1][1]+m[0][1])))
        summary.loc['NPV'][hp, key] = float(m[1][1]/(m[1][1]+m[0][1]))
        # print("Sensitivity:", recall_score(y_test-1, results[key][1]-1))
        summary.loc['Sensitivity'][hp, key] = recall_score(y_test-1, results[key][1]-1)
        lower, upper = proportion_confint(m[0][0], m[0][0] + m[0][1], 0.05, method='normal')
        # print("Sensitivity 95% confidence interval:", lower, upper)
        summary.loc['Sensitivity-95%-CI-low'][hp, key] = lower
        summary.loc['Sensitivity-95%-CI-up'][hp, key] = upper
        # print('Specificity:', float(m[1][1]/(m[1][0]+m[1][1])))
        summary.loc['Specificity'][hp, key] = float(m[1][1]/(m[1][0]+m[1][1]))
        lower, upper = proportion_confint(m[1][1], m[1][0] + m[1][1], 0.05, method='normal')
        # print("Specificity 95% confidence interval:", lower, upper)
        summary.loc['Specificity-95%-CI-low'][hp, key] = lower
        summary.loc['Specificity-95%-CI-up'][hp, key] = upper
        fpr, tpr, thr_ = roc_curve(y_test, results[key][2].T[1], pos_label=2)
        area = auc(fpr, tpr)
        # print("AUC:", area)
        summary.loc['AUC'][hp, key] = area

        def se_auc(area, n1, n2):
            q1 = area / (2 - area)
            q2 = 2 * area ** 2 / (1 + area)
            return np.sqrt((area * (1 - area) + (n1 - 1) * (q1 - area ** 2) + (n2 - 1) * (q2 - area ** 2)) / (n1 * n2))
        se = se_auc(area, m[0][0]+m[0][1], m[1][0]+m[1][1])
        # print("AUC 95% confidence interval:", area-1.96*se, min(area+1.96*se, 1.0))
        summary.loc['AUC-95%-CI-low'][hp, key] = area-1.96*se
        summary.loc['AUC-95%-CI-up'][hp, key] = min(area+1.96*se, 1.0)

        print("####################################################")

    # sys.stdout = old  # 还原系统输出
    # file.close()
    summary.to_excel(new_path+'/'+mode+'-'+str(len(results))+'-summary.xlsx')

    # 绘制ROC、PRC、混淆矩阵, calibration curve/density
    labels_roc = []
    labels_prc = []
    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'seagreen', 'pink']
    plt.style.use("ggplot")
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10), dpi=400)
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10), dpi=400)
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 10), dpi=400)
    ax3.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 5), dpi=400)
    for i, (key, item) in zip(range(6), results.items()):
        p, r, thr = precision_recall_curve(y_test, results[key][2].T[1], pos_label=2)
        fpr, tpr, thr_ = roc_curve(y_test, results[key][2].T[1], pos_label=2)
        labels_roc.append('area of ROC for '+key+' = {0:0.4f}'
                          ''.format(auc(fpr, tpr)))
        labels_prc.append('area of PRC for '+key+' = {0:0.4f}'
                          ''.format(average_precision_score(y_test, results[key][2].T[1], pos_label=2)))
        ax1.step(fpr, tpr, where='post', lw=2, color=colors[i])
        ax2.step(r, p, where='post', lw=2, color=colors[i])

        # draw calibration curve
        prob_pos = item[2].T[1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=5)
        ax3.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (key, ))
        ax4.hist(prob_pos, label=key, histtype='step', lw=2)

    ax1.legend(labels_roc, loc='lower right', prop=dict(size=14))
    ax1.set_title('Receiver Operating Characteristic curve', fontsize=14)
    ax1.set_xlabel('FPR', fontsize=14)
    ax1.set_ylabel('TPR', fontsize=14)
    ax1.grid()
    ax2.legend(labels_prc, loc='lower left', prop=dict(size=14))
    ax2.set_title('Precision-Recall curve', fontsize=14)
    ax2.set_xlabel('Recall', fontsize=14)
    ax2.set_ylabel('Precision', fontsize=14)
    ax2.grid()

    ax3.set_ylabel("Fraction of positives")
    ax3.set_ylim([-0.05, 1.05])
    ax3.legend(loc="lower right")
    ax3.set_title('Calibration plots  (reliability curve)')

    ax4.set_xlabel("Mean predicted value")
    ax4.set_ylabel("Count")
    ax4.legend(loc="upper center", ncol=2)
    # save figures
    figs = [fig1, fig2, fig3, fig4]
    figs_name = ['-roc', '-prc', '-cal', '-hist']
    # path = './results/' + dt.datetime.now().strftime('%Y%m%d-%H%M')+'-'+hp
    for i, fig in enumerate(figs):
        fig_name = figs_name[i]
        fig.tight_layout()
        fig.savefig(new_path+'/'+hp+'-'+fig_name + ".pdf", dpi=400)
        '''


def plot_result(feature_importance):
    k = 1
    x1 = []
    cnt = 0
    while cnt < len(feature_importance):
        x1.append(k)
        k = k + 4
        cnt = cnt + 1

    x = range(len(feature_importance))
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    ax[0].bar(x, feature_importance[0], color='green', label="gbdt")
    ax[1].bar(x, feature_importance[1], color='red', label="rf")
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xticks(x)
    label = columns
    ax[1].set_xticklabels(label)
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(90)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    y_train = y_train.reshape(len(y_train), )
    X_test = [test_data_gg, X_test_zf, test_data_xy]
    y_test = [test_label_gg.reshape(len(test_label_gg), ),
              y_test_zf.reshape(len(y_test_zf), ),
              test_label_xy.reshape(len(test_data_xy), )]
    patient_id = [id_gg, id_zf, id_xy]
    for i, hp in enumerate(['gg']):  # , 'zf', 'xy']):
        rf_results = rf(X_train, y_train, X_test[i], y_test[i])
        gbdt_results = gbdt(X_train, y_train, X_test[i], y_test[i])
        lr_results = lr(X_train, y_train, X_test[i], y_test[i])
        knn_results = knn(X_train, y_train, X_test[i], y_test[i])
        svm_results = svm(X_train, y_train, X_test[i], y_test[i])
        stack_results = stack_models(X_train, y_train, X_test[i], y_test[i])
        weights = {'gg': np.array([1, 0.7, 1]),
                   'zf': np.array([2, 2, 0]),
                   'xy': np.array([4, 4, 0])}
        vote_results = voting(rf_results[2],
                              gbdt_results[2],
                              lr_results[2],
                              svm_results[2],
                              np.array([1, 1, 1, 1]))
        predict = np.hstack((stack_results[1].reshape(len(vote_results[1]), 1), patient_id[i]))
        predict_res = pd.DataFrame(predict, columns=['label', 'id'])
        # path = './results/' + dt.datetime.now().strftime('%Y%m%d-%H-%M')
        predict_res.to_excel(path+'/'+hp+'-'+mode+'-predict-results.xlsx', index=False)
        results1 = {'EmsembleModel': stack_results,
                    'RF': rf_results,
                    'GBDT': gbdt_results,
                    'LR': lr_results}
        results2 = {
                   'SVM': svm_results,
                   'KNN': knn_results}
        analysis_results(results1, X_test[i], y_test[i], hp)
        analysis_results(results2, X_test[i], y_test[i], hp)

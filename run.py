from data_preprocess import generate_data, generate_train_data
import pandas as pd
from Model import RF, GBDT, LRl2, LRl1, KNN, SVM, MLP, StackModel, VotingModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, \
    f1_score, roc_curve, average_precision_score, auc, confusion_matrix, precision_score, recall_score
import datetime as dt
import os
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint
from sklearn.calibration import calibration_curve
from sklearn.externals import joblib
import shap_interpretation

X_train, y_train, X_test_zf, y_test_zf, id_zf = generate_train_data('./data_filter30_svdimpute/SF.xlsx',
                                                                    delete_n_last_features=False,
                                                                    over_sample=False)
test_data_gg, test_label_gg, columns, id_gg = generate_data('./data_filter30_svdimpute/OV.xlsx',
                                                            delete_n_last_features=False)
test_data_xy, test_label_xy, _, id_xy = generate_data('./data_filter30_svdimpute/CHWH.xlsx',
                                                      delete_n_last_features=False)

# 结果路径
path = './results/'+dt.datetime.now().strftime('%Y%m%d-%H-%M')
os.makedirs(path, exist_ok=True)
# 保存模型路径
save_models = './save_models/' + dt.datetime.now().strftime('%Y%m%d-%H-%M')
os.makedirs(save_models, exist_ok=True)




def gbdt(X_train, y_train, X_test, y_test):
    # GBDT
    gbdt = GBDT(X_train, y_train, X_test, y_test)
    # train
    gbdt_feature_importance, estimator_score = gbdt.gbdt_train()
    # save model
    joblib.dump(gbdt.gbdt, save_models+'/gbdt.pkl')
    # predict
    gbdt_predict_result = gbdt.gbdt_predict()
    # predict probability
    gbdt_predict_result_prob = gbdt.gbdt_predict_prob()
    # get score
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
    # train
    rf_feature_importance, oob_score = rf.rf_train()
    # save model
    joblib.dump(rf.rf, save_models+'/rf.pkl')
    # predict
    rf_predict_result = rf.rf_predict()
    # predict probability
    rf_predict_result_prob = rf.rf_predict_prob()
    # get score
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


def lrl2(X_train, y_train, X_test, y_test):
    # lr
    lr = LRl2(X_train, y_train, X_test, y_test)
    # train
    lr_coef_ = lr.lr_train()
    # save model
    joblib.dump(lr.lr, save_models+'/lrl2.pkl')
    # predict
    lr_predict = lr.lr_predict()
    # predict probability
    lr_predict_proba = lr.lr_predict_proba()

    # save rf feature importance
    df_lr_feature_importance = pd.DataFrame(data=np.abs(lr_coef_),
                                            columns=list(columns))
    df_lr_feature_importance.sort_values(by=0, axis=1, ascending=False, inplace=True)
    # df_lr_feature_importance.loc[0, df_lr_feature_importance.columns.values].plot(kind='bar')
    # path = './feature_select/' + dt.datetime.now().strftime('%Y%m%d-%H-%M')
    df_lr_feature_importance.to_excel(path+'/'+'lrl2_feature_importance.xlsx', index=False)
    print("lr L2 get score:", lr.lr_score())
    return [lr, lr_predict, lr_predict_proba]


def lrl1(X_train, y_train, X_test, y_test):
    # lr
    lr = LRl1(X_train, y_train, X_test, y_test)
    # train
    lr_coef_ = lr.lr_train()
    # save model
    joblib.dump(lr.lr, save_models+'/lrl1.pkl')
    # predict
    lr_predict = lr.lr_predict()
    # predict probability
    lr_predict_proba = lr.lr_predict_proba()

    # save rf feature importance
    df_lr_feature_importance = pd.DataFrame(data=np.abs(lr_coef_),
                                            columns=list(columns))
    df_lr_feature_importance.sort_values(by=0, axis=1, ascending=False, inplace=True)
    # df_lr_feature_importance.loc[0, df_lr_feature_importance.columns.values].plot(kind='bar')
    # path = './feature_select/' + dt.datetime.now().strftime('%Y%m%d-%H-%M')
    df_lr_feature_importance.to_excel(path+'/'+'lrl1_feature_importance.xlsx', index=False)
    print("lr L1 get score:", lr.lr_score())
    return [lr, lr_predict, lr_predict_proba]


def knn(X_train, y_train, X_test, y_test):
    # knn
    knn = KNN(X_train, y_train, X_test, y_test)
    # train
    knn.knn_train()
    # save model
    joblib.dump(knn.knn, save_models+'/knn.pkl')
    # predict
    knn_predict = knn.knn_predict()
    # predict probability
    knn_predict_proba = knn.knn_predict_proba()
    print("knn get score:", knn.score())
    return [knn, knn_predict, knn_predict_proba]


def svm(X_train, y_train, X_test, y_test):
    # svc
    svm = SVM(X_train, y_train, X_test, y_test)
    # train
    svm_coef_ = svm.svm_train()
    # save model
    joblib.dump(svm.svm, save_models+'/svm.pkl')
    # predict
    svm_predict = svm.svm_predict()
    # predict confidence
    svm_confidence = svm.svm_predict_confidence()
    df_svm_feature_importance = pd.DataFrame(data=np.abs(svm_coef_),
                                             columns=list(columns))
    df_svm_feature_importance.sort_values(by=0, axis=1, ascending=False, inplace=True)
    # df_svm_feature_importance.loc[0, df_svm_feature_importance.columns.values].plot(kind='bar')
    # path = './feature_select/' + dt.datetime.now().strftime('%Y%m%d-%H-%M')
    df_svm_feature_importance.to_excel(path+'/'+'svm_feature_importance.xlsx', index=False)

    print("svm get score:", svm.svm_score())
    return [svm, svm_predict, svm_confidence]


def mlp(X_train, y_train, X_test, y_test):
    # mlp
    mlp = MLP(X_train, y_train, X_test, y_test)
    # mlp train
    mlp.mlp_train()
    # save model
    joblib.dump(mlp.mlp, save_models+'/mlp.pkl')
    # predict
    mlp_predict = mlp.mlp_predict()
    # predict probability
    mlp_predict_prob = mlp.mlp_predict_prob()

    print("MLP get score:", mlp.mlp_score())

    return [mlp, mlp_predict, mlp_predict_prob]


def stack_models(X_train, y_train, X_test, y_test):
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
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


def voting(rf, gbdt, lr, svm, weights):  #svm,
    voting_proba = np.zeros((len(rf), 2))
    voting_label = np.zeros((len(rf), ))
    all_proba = np.hstack((rf[:, 1].reshape(len(rf), 1),
                           # gbdt[:, 1].reshape(len(rf), 1),
                           lr[:, 1].reshape(len(rf), 1),
                           svm[:, 1].reshape(len(svm), 1)))

    for i in range(len(all_proba)):
        voting_proba[i][1] = (all_proba[i][0]*weights[0] + all_proba[i][1]*weights[1] +
                              all_proba[i][2]*weights[2])/np.sum(weights)  #
        voting_proba[i][0] = 1 - voting_proba[i][1]
        if voting_proba[i][1] >= 0.5:
            voting_label[i] = 2
        else:
            voting_label[i] = 1

    return [rf, voting_label, voting_proba]


summary1 = pd.DataFrame(data=np.zeros((19, 27)),
                        columns=pd.MultiIndex.from_product([['gg', 'zf', 'xy'],
                                                            ['EmsembleModel', 'Voting', 'RF', 'GBDT', 'LRl2', 'LRL1', 'SVM', 'MLP', 'KNN']]),
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
                              'TN']
                       )


summary2 = pd.DataFrame(data=np.zeros((19, 9)),
                        columns=pd.MultiIndex.from_product([['gg', 'zf', 'xy'],
                                                           ['MLP', 'SVM', 'KNN']]),
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
                               'TN']
                       )


def analysis_results(results, y_test, hp):
    # file = open(path+'/'+hp+'-summary.txt', 'w')
    # old = sys.stdout
    # sys.stdout = file
    file = str(len(results))
    new_path = path+'/'+file+'/'
    os.makedirs(new_path, exist_ok=True)
    if len(results) == 9:
        summary = summary1
    else:
        summary = summary2
    shap_interpretation.run_shap(results, X_test, new_path, hp)

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
    summary.to_excel(new_path+'/'+'-'+str(len(results))+'-summary.xlsx')

    # 绘制ROC、PRC、混淆矩阵
    labels_roc = []
    labels_prc = []
    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'seagreen', 'pink', 'red', 'green', 'orange']
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


if __name__ == "__main__":

    y_train = y_train.reshape(len(y_train), )
    X_test = [test_data_gg, X_test_zf, test_data_xy]
    y_test = [test_label_gg.reshape(len(test_label_gg), ),
              y_test_zf.reshape(len(y_test_zf), ),
              test_label_xy.reshape(len(test_data_xy), )]
    patient_id = [id_gg, id_zf, id_xy]
    for i, hp in zip(range(3), ['gg', 'zf', 'xy']):
        rf_results = rf(X_train, y_train, X_test[i], y_test[i])
        gbdt_results = gbdt(X_train, y_train, X_test[i], y_test[i])
        lrl2_results = lrl2(X_train, y_train, X_test[i], y_test[i])
        lrl1_results = lrl1(X_train, y_train, X_test[i], y_test[i])
        knn_results = knn(X_train, y_train, X_test[i], y_test[i])
        svm_results = svm(X_train, y_train, X_test[i], y_test[i])
        mlp_results = mlp(X_train, y_train, X_test[i], y_test[i])
        # stack method
        stack_results = stack_models(X_train, y_train, X_test[i], y_test[i])
        # voting method
        weights = {'gg': np.array([1, 0.7, 1]),
                   'zf': np.array([2, 2, 0]),
                   'xy': np.array([4, 4, 0])}
        vote_results = voting(rf_results[2],
                              lrl2_results[2],
                              svm_results[2],
                              gbdt_results[2],
                              np.array([1.3, 1.2, 0.8]))
        predict = np.hstack((stack_results[1].reshape(len(stack_results[1]), 1), patient_id[i]))
        predict_res = pd.DataFrame(predict, columns=['label', 'id'])
        predict_res.to_excel(path+'/'+hp+'-'+'-predict-results.xlsx', index=False)
        results1 = {'EmsembleModel': stack_results,
                    'Voting': vote_results,
                    'RF': rf_results,
                    'GBDT': gbdt_results,
                    'LRl2': lrl2_results,
                    'LRL1': lrl1_results,
                    'SVM': svm_results,
                    'MLP': mlp_results,
                    'KNN': knn_results
                    }
        '''
        results2 = {
                   'MLP': mlp_results,
                   'SVM': svm_results,
                   'KNN': knn_results}
                   '''
        analysis_results(results1, y_test[i], hp)
        # analysis_results(results2, y_test[i], hp)

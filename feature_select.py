import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_preprocess import generate_data, generate_train_data
from run import rf, lrl2, svm, voting
from sklearn.metrics import roc_curve, auc
import datetime as dt


def combine_features_rank(feature_path):
    '''
    对 rf,gbdt, lr, svm的排序结果进行加权平均，
    :param delete_grade:
    :return:
    '''
    model_feature = {}
    model_list = ['rf', 'lrl2', 'svm']
    for model in model_list:
        path = feature_path+model+'_feature_importance.xlsx'
        columns = pd.read_excel(path).columns.values.tolist()
        index = [i+1 for i in range(len(columns))]
        model_feature[model] = dict(zip(columns, index))
    # 每种模型排名对应的权重，用对应的F1值表示
    weights = {'rf': 0.7, 'lrl2': 0.72, 'svm': 0.73}
    columns = list(model_feature['rf'].keys())
    new_rank = {}
    for col in columns:
        s = 0
        for key in model_list:
            s = s + model_feature[key][col] * weights[key]
        new_rank[col] = float(1.0/float(s/3.0))
    new_rank_df = pd.DataFrame(new_rank, index=[0])
    new_rank_df.sort_values(by=0, axis=1, ascending=False, inplace=True)
    new_rank_df.to_excel('./feature_select/combine_feature_rank.xlsx', index=False)
    new_rank_df.loc[0, new_rank_df.columns.values].plot(kind='bar')
    plt.show(block=False)
    return new_rank_df


def run(res_auc,
        delete_n_last_common_features,
        n_common_last,
        delete_n_last_features,
        n_last):
    X_train, y_train, X_test_zf, y_test_zf, id_zf = generate_train_data('./data_filter30_svdimpute/SF.xlsx',
                                                                        delete_n_last_common_features=delete_n_last_common_features,
                                                                        n_common_last=n_common_last,
                                                                        delete_n_last_features=delete_n_last_features,
                                                                        n_last=n_last)
    test_data_gg, test_label_gg, columns, id_gg = generate_data('./data_filter30_svdimpute/OV.xlsx',
                                                                delete_n_last_common_features=delete_n_last_common_features,
                                                                n_common_last=n_common_last,
                                                                delete_n_last_features=delete_n_last_features,
                                                                n_last=n_last)
    test_data_xy, test_label_xy, _, id_xy = generate_data('./data_filter30_svdimpute/CHWH.xlsx',
                                                          delete_n_last_common_features=delete_n_last_common_features,
                                                          n_common_last=n_common_last,
                                                          delete_n_last_features=delete_n_last_features,
                                                          n_last=n_last)

    y_train = y_train.reshape(len(y_train), )
    X_test = [test_data_gg, X_test_zf, test_data_xy]
    y_test = [test_label_gg.reshape(len(test_label_gg), ),
              y_test_zf.reshape(len(y_test_zf), ),
              test_label_xy.reshape(len(test_data_xy), )]
    n = 0
    if delete_n_last_common_features:
        n = n_common_last
    if delete_n_last_features:
        n = n_last
    res_auc.loc[n]['feature_num'] = X_train.shape[1]
    for i, hp in zip(range(3), ['gg', 'zf', 'xy']):
        rf_results = rf(X_train, y_train, X_test[i], y_test[i])
        lrl2_results = lrl2(X_train, y_train, X_test[i], y_test[i])
        svm_results = svm(X_train, y_train, X_test[i], y_test[i])
        # voting method
        vote_results = voting(rf_results[2],
                              lrl2_results[2],
                              svm_results[2],
                              np.array([1.2, 1.2, 1.1]))
        fpr, tpr, thr_ = roc_curve(y_test[i], vote_results[2].T[1], pos_label=2)
        res_auc.loc[n][hp] = auc(fpr, tpr)


def feature_select_by_delete_common_features():
    # method 1
    n_last_lists = [5, 10, 15, 20, 25, 30]
    res_auc = pd.DataFrame(data=np.zeros((len(n_last_lists), 4)),
                           columns=['feature_num', 'gg', 'zf', 'xy'],
                           index=n_last_lists)
    for i in n_last_lists:
        run(res_auc,
            True,
            i,
            False,
            0)
    res_auc.set_index('feature_num', drop=True, inplace=True)
    res_auc.to_excel('./feature_select/VoteModel-AUC-Vary-with-Feature-Num-1.xlsx')
    ax = res_auc.plot(kind='bar')
    fig = ax.get_figure()
    fig.savefig('./feature_select/method1.pdf', dpi=400)


def feature_select_by_delete_combine_features():
    # method 2
    n_last_features = [i for i in range(1, 20, 2)]
    combine_features_rank('./feature_select/')
    res_auc = pd.DataFrame(data=np.zeros((len(n_last_features), 4)),
                           columns=['feature_num', 'gg', 'zf', 'xy'],
                           index=n_last_features)
    for i in n_last_features:
        run(res_auc,
            False,
            0,
            True,
            i)
    res_auc.set_index('feature_num', drop=True, inplace=True)
    res_auc.to_excel('./feature_select/VoteModel-AUC-Vary-with-Feature-Num-2'
                     + dt.datetime.now().strftime('%Y%m%d-%H-%M')+'.xlsx')
    ax = res_auc.plot(kind='bar')
    fig = ax.get_figure()
    fig.savefig('./feature_select/method2.pdf', dpi=400)


if __name__ == "__main__":
    # feature_select_by_delete_common_features()
    feature_select_by_delete_combine_features()

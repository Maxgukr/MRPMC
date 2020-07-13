import pandas as pd
import json
from numpy.random import uniform
import numpy as np
import math
from sklearn.utils import shuffle
from collections import Counter


def load_data(file_name, feature_map):
    print("loading data...")
    fm = json.load(open(feature_map, 'r'))
    data = pd.read_excel(file_name, dtype={'Name': str, 'Value': float})
    # data.drop(columns=['ID'], inplace=True)
    # data['Death'] = data['Death'] - 1
    columns = data.columns.values
    for col in columns:
        if 'Grade1' in col:
            data.drop(columns=[col], inplace=True)
    data.drop(columns=['Nucleic1'], inplace=True)
    print("loading data finished!")
    return data, fm


def fill_nan(data, fm, filename):
    """
    fill Nan value by normal range
    :param data: input data
    :return: output data
    """
    print("fill Nan begining...")

    # fill categorical variable
    categorical_variable_fill = pd.read_excel("./data/VariableFill.xlsx", dtype={'Name': str, 'Value': float})
    categorical_features = categorical_variable_fill['Variable'].values
    categorical_variable_fill.set_index('Variable', inplace=True)
    for cat_col in categorical_features:
        data[cat_col].fillna(categorical_variable_fill.loc[cat_col, 'Fill'], inplace=True)

    # fill continue variable
    columns_name = list(data.columns.values)
    feature_name = list(fm.keys())

    for col in columns_name:
        if col in feature_name:
            for i in range(len(data)):
                # 填充空值
                if np.isnan(data.loc[i, col]):
                    if len(fm[col]) == 4:
                        data.loc[i, col] = uniform(fm[col]['low'], fm[col]['up'], 1)
                        data.loc[i, col+'Grade2'] = fm[col]['grade2']
                    else:
                        if data.loc[i, 'Gender'] == 1:
                            data.loc[i, col] = uniform(fm[col]['F']['low'], fm[col]['F']['up'], 1)
                            data.loc[i, col+'Grade2'] = fm[col]['F']['grade2']
                        elif data.loc[i, 'Gender'] == 0:
                            data.loc[i, col] = uniform(fm[col]['M']['low'], fm[col]['M']['up'], 1)
                            data.loc[i, col + 'Grade2'] = fm[col]['M']['grade2']

    data.to_csv('./data/'+filename+'processed_data.csv', index=False)
    print("fill Nan finished! processed data is saved in ./data/processed_data.csv")
    return data


def delete_last_features(rf, gbdt, lr, svm, n_last=20, thr=2):
    '''
    :param rf: rf feature rank
    :param gbdt: gbdt feature rank
    :param lr: lr feature rank
    :param svm: svm feature rank
    :param n_last: The number of times the feature appears in the last
    :param thr: The number of times the feature appears in the last "n_last"，default is 3
    :return:
    '''
    rf = pd.read_excel(rf).columns.values.tolist()
    lr = pd.read_excel(lr).columns.values.tolist()
    svm = pd.read_excel(svm).columns.values.tolist()
    if gbdt is None:
        n_last_list = rf[-n_last:] + lr[-n_last:] + svm[-n_last:]
    else:
        gbdt = pd.read_excel(gbdt).columns.values.tolist()
        n_last_list = rf[-n_last:] + lr[-n_last:] + svm[-n_last:] + gbdt[-n_last:]
    feature_count = dict(Counter(n_last_list))
    n_last_delete = []
    for key, value in feature_count.items():
        if value >= thr:
            n_last_delete.append(key)

    return n_last_delete


def generate_train_data(filename,
                        split_rate=0.8,
                        delete_n_last_common_features=False,
                        n_common_last=20,
                        delete_n_last_features=False,
                        n_last=0):
    '''
    deal with SF data set, split for train and test data
    :param filename: SF data set file path
    :param split_rate: split rate, default is 0.8
    :param delete_n_last_common_features: delete method 1
    :param n_common_last: delete number
    :param delete_n_last_features delete method 2
    :param n_last delete number
    :return: x_tarin, y_train, x_test, y_test
    '''
    data = pd.read_excel(filename)
    # delete n_last features from combine feature rank from rf, lr, svm
    if delete_n_last_features:
        combine_feature_rank = pd.read_excel('./feature_select/combine_feature_rank.xlsx')
        features = combine_feature_rank.columns.values[-n_last:]
        data.drop(columns=features, inplace=True)

    # delete n_common_last features in features rank of rf, ls, svm
    if delete_n_last_common_features:
        path = './feature_select/'
        n_last_delete = delete_last_features(rf=path + 'rf_feature_importance.xlsx',
                                             lr=path + 'lrl2_feature_importance.xlsx',
                                             svm=path + 'svm_feature_importance.xlsx',
                                             n_last=n_common_last)
        data.drop(columns=n_last_delete, inplace=True)

    df_death = data.loc[data.Death == 2]
    df_live = data.loc[data.Death == 1]

    # When splitting, separate death and discharge first to avoid uneven splitting
    death_split_index = math.ceil(len(df_death) * split_rate)

    df_death_train = df_death[:death_split_index]
    df_death_test = df_death[death_split_index:]

    live_split_index = math.ceil(len(df_live) * split_rate)

    df_live_train = df_live[:live_split_index]
    df_live_test = df_live[live_split_index:]

    # Combine the death data needed for training with the discharge data for training and then break up the order
    df_train = pd.concat([df_death_train, df_live_train], axis=0)
    df_train = shuffle(df_train)
    # Combine the death data required by test with the discharge data and break up the order
    df_test = pd.concat([df_death_test, df_live_test], axis=0)
    df_test = shuffle(df_test)

    # get train and test label y_train, y_test
    y_train = df_train.get(['Death']).values.reshape(len(df_train), 1)
    y_test = df_test.get(['Death']).values.reshape(len(df_test), 1)
    df_train.drop(columns=['Death', 'ID'], inplace=True)
    id1 = df_test.get(['ID']).values.reshape(len(df_test), 1)

    # delete Death and ID item in test data
    df_test.drop(columns=['Death', 'ID'], inplace=True)

    return df_train, y_train, df_test, y_test, id1


def generate_data(filename,
                  delete_n_last_common_features=False,
                  n_common_last=20,
                  delete_n_last_features=False,
                  n_last=0):
    '''
    deal with OV and CHWH dataset
    :param filename:
    :param delete_n_last_common_features: delete method 1
    :param n_common_last: delete number
    :param delete_n_last_features delete method 2
    :param n_last delete number
    :return: x_tarin, y_train, x_test, y_test
    '''
    data = pd.read_excel(filename)
    if delete_n_last_features:
        combine_feature_rank = pd.read_excel('./feature_select/combine_feature_rank.xlsx')
        features = combine_feature_rank.columns.values[-n_last:]
        data.drop(columns=features, inplace=True)
    # delete last n features in features rank of rf, gbdt, ls, svm
    if delete_n_last_common_features:
        path = './feature_select/'
        n_last_delete = delete_last_features(rf=path+'rf_feature_importance.xlsx',
                                             lr=path+'lrl2_feature_importance.xlsx',
                                             svm=path+'svm_feature_importance.xlsx',
                                             n_last=n_common_last)
        data.drop(columns=n_last_delete, inplace=True)

    index = data.index[np.where(np.isnan(data))[0]].values
    assert len(index) == 0

    ID = data.get(['ID']).values.reshape(len(data), 1)
    data.drop(columns=['ID'], inplace=True)

    columns = data.columns.values[:-1]
    label = data.get(['Death']).values.reshape(len(data), 1)
    data.drop(columns=['Death'], inplace=True)
    return data, label, columns, ID


def data_description(filename):
    '''
    :param filename:
    :return:
    '''
    data = pd.read_csv(filename, dtype={'Name': str, 'Value': float})
    delete_features = ['SOFA score', 'Corticosteroids', 'DurationCorticosteroids',
                       'Intravenous immunoglobin', 'Antibiotics',
                       'carbostyril', 'cephalosporin', 'broad_spectrum',
                       'Vasoactive', 'DurationVasoactive', 'Lac', 'LacGrade2',
                       'Ribavirin', 'Oxygen therapy', 'DurationInterferon',
                       'Oseltamivir', 'Arbidol', 'Lopinavir',
                       'Detection approach', 'DurationRibavirin', 'DurationLopinavir',
                       'DurationOseltamivir', 'DurationArbidol', 'IgM', 'IgG', 'IgGGrade2',
                       'Oxygen therapy approach', 'IgMGrade2', 'Interferon', 'ID']
    data.drop(columns=delete_features, inplace=True)
    columns = data.columns.values
    for col in columns:
        if 'Grade' in col:
            data.drop(columns=[col], inplace=True)
    path = './feature_select/20200509-15-58-'
    n_last_delete = delete_last_features(rf=path+'rf_feature_importance.xlsx',
                                         gbdt=path+'gbdt_feature_importance.xlsx',
                                         lr=path+'lr_feature_importance.xlsx',
                                         svm=path+'svm_feature_importance.xlsx',
                                         n_last=20,
                                         thr=3)
    data.drop(columns=n_last_delete, inplace=True)
    data.drop(columns=['Death'], inplace=True)
    data.to_excel('./feature_select/impute_with_normal_range_'+filename.split('/')[-1][:2]+'.xlsx', index=False)


if __name__ == "__main__":
    # combine_features_rank()
    # data, fm = load_data('./data/COVID-19-xy.xlsx', './data/feature-map.json')
    # test_data_xy, test_label_xy, _, id_xy = generate_data('./data/COVID-19-xy.xlsx')
    # data = fill_nan(data, fm, 'xy')
    data_description('./data_impute_with_normal_range/xyprocessed_data.csv')
    data_description('./data_impute_with_normal_range/zfprocessed_data.csv')
    data_description('./data_impute_with_normal_range/ggprocessed_data.csv')
    print("processed finished!")

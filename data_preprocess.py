import pandas as pd
import json
from numpy.random import uniform
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.utils import shuffle
from collections import Counter
import datetime as dt
from imblearn.over_sampling import SMOTENC


# 这个文件不用管
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


# 这个文件不用管
def fill_nan(data, fm, filename):
    """
    fill Nan value
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


def delete_last_features(rf, gbdt, lr, svm, n_last=20, thr=3):
    '''
    :param rf: rf特征排序结果，以下含义类推
    :param gbdt:
    :param lr:
    :param svm:
    :param n_last: 排名倒数多少个，默认倒数20个
    :param thr: 出现次数的阈值，默认为3
    :return:
    '''
    rf = pd.read_excel(rf).columns.values.tolist()
    gbdt = pd.read_excel(gbdt).columns.values.tolist()
    lr = pd.read_excel(lr).columns.values.tolist()
    svm = pd.read_excel(svm).columns.values.tolist()
    n_last_list = rf[-n_last:] + gbdt[-n_last:] + lr[-n_last:] + svm[-n_last:]
    feature_count = dict(Counter(n_last_list))
    n_last_delete = []
    for key, value in feature_count.items():
        if value >= thr:
            n_last_delete.append(key)

    return n_last_delete


def generate_train_data(filename, split_rate=0.8, delete_grade=False, over_sample=False):
    '''
    专门处理中法医院的数据，中法医院一部分用作训练集，一部分用作内部测试集
    :param filename: 中法医院数据文件
    :param split_rate: 分割比例
    :param delete_grade: 是否删除Grade2特征
    :param over_sample: 是否采用SMOTENC算法对死亡样本进行过采样
    :return: x_tarin, y_train, x_test, y_test
    '''
    data = pd.read_csv(filename)

    # 删除特征分成了三步
    # 1.
    # delete features without used
    # delete because of Clinical significance 从临床意义上删除以下特征
    delete_features = ['SOFA score', 'Corticosteroids', 'DurationCorticosteroids',
                       'Intravenous immunoglobin', 'Antibiotics',
                       'carbostyril', 'cephalosporin', 'broad_spectrum',
                       'Vasoactive', 'DurationVasoactive', 'Lac', 'LacGrade2',
                       'Ribavirin', 'Oxygen therapy', 'DurationInterferon',
                       'Oseltamivir', 'Arbidol', 'Lopinavir',
                       'Detection approach', 'DurationRibavirin', 'DurationLopinavir',
                       'DurationOseltamivir', 'DurationArbidol', 'IgM', 'IgG', 'IgGGrade2', 'IgMGrade2',
                       'Oxygen therapy approach', 'Interferon']
    data.drop(columns=delete_features, inplace=True)

    # 2.
    # delete Grade2 删除人为创造的分级特征Grade2
    if delete_grade:
        columns = data.columns.values
        for col in columns:
            if 'Grade2' in col:
                data.drop(columns=[col], inplace=True)

    # 3.
    # 删除特征排序中，在RF，GBDT，LR，SVM中排名最后20个中，出现三次以上的特征
    # delete last 20 features in features rank of rf,gbdt,ls, svm
    path = './feature_select/' + '20200509-15-58-'
    n_last_delete = delete_last_features(path + 'rf_feature_importance.xlsx',
                                         path + 'gbdt_feature_importance.xlsx',
                                         path + 'lr_feature_importance.xlsx',
                                         path + 'svm_feature_importance.xlsx',
                                         20,
                                         3)
    data.drop(columns=n_last_delete, inplace=True)
    '''
    file = open('./feature_select/features.txt', 'w')
    for f in data.columns.values.tolist():
        file.write(f)
        file.write('\n')
    file.close()
    '''

    data.to_excel(filename + '-COVID_FillNan_delete_features.xlsx', index=False)
    df_death = data.loc[data.Death == 2]
    df_live = data.loc[data.Death == 1]

    # split zhong-fa hospital data split的时候，将死亡和出院先分开，以免分割不均匀
    death_split_index = math.ceil(len(df_death) * split_rate)

    df_death_train = df_death[:death_split_index]
    df_death_test = df_death[death_split_index:]

    live_split_index = math.ceil(len(df_live) * split_rate)

    df_live_train = df_live[:live_split_index]
    df_live_test = df_live[live_split_index:]

    # 将train需要的死亡数据与训练用的出院数据合并再打散顺序
    df_train = pd.concat([df_death_train, df_live_train], axis=0)
    df_train = shuffle(df_train)
    # 将test需要的死亡数据与训练用的出院数据合并再打散顺序
    df_test = pd.concat([df_death_test, df_live_test], axis=0)
    df_test = shuffle(df_test)

    # 获得train数据和test数据的label y_train, y_test
    y_train = df_train.get(['Death']).values.reshape(len(df_train), 1)
    y_test = df_test.get(['Death']).values.reshape(len(df_test), 1)
    df_train.drop(columns=['Death', 'ID'], inplace=True)
    id1 = df_test.get(['ID']).values.reshape(len(df_test), 1)

    # 获得train数据和test数据的输入，x_train，x_test
    df_test.to_excel('./data_description/'+dt.datetime.now().strftime('%Y%m%d-%H-%M')+'-zf_test.xlsx', index=False)
    df_test.drop(columns=['Death', 'ID'], inplace=True)
    x_train = df_train.values
    x_test = df_test.values

    # using over sampling or not for few label
    # 是否采用SMOTENC算法对样本较少的死亡样本进行人工扩充（生成伪死亡样本）
    if over_sample:
        # print(Counter(y_train.reshape(len(y_train, )).tolist()))
        # using SOMTENC for over sample
        smo = SMOTENC(categorical_features=[1, 2, 3, 7, 8, 9, 10, 38, 39],
                      sampling_strategy=0.5,
                      random_state=42)
        X_train, y_train = smo.fit_sample(x_train, y_train)
        # print(Counter(y_smo.reshape(len(y_smo, )).tolist()))
        df_train_smo = pd.DataFrame(data=np.hstack((X_train, y_train.reshape(len(y_train), 1))),
                                    columns=data.drop(columns=['ID']).columns.values.tolist())
        df_train_smo = shuffle(df_train_smo)
        y_train = df_train_smo.get(['Death']).values
        df_train_smo.drop(columns=['Death'], inplace=True)
        x_train = df_train_smo.values

    return x_train, y_train, x_test, y_test, id1


def generate_data(filename, delete_grade=False):
    '''
    用来处理光谷院区，襄阳院区的数据
    :param filename: 医院数据文件
    :param delete_grade: 是否删除Grade2特征
    :return: x_tarin, y_train, x_test, y_test
    '''
    data = pd.read_csv(filename)

    # 删除特征分成了三步
    # 1.
    # delete features without used
    # delete because of Clinical significance 从临床意义上删除以下特征
    delete_features = ['SOFA score', 'Corticosteroids', 'DurationCorticosteroids',
                       'Intravenous immunoglobin', 'Antibiotics',
                       'carbostyril', 'cephalosporin', 'broad_spectrum',
                       'Vasoactive', 'DurationVasoactive', 'Lac', 'LacGrade2',
                       'Ribavirin', 'Oxygen therapy', 'DurationInterferon',
                       'Oseltamivir', 'Arbidol', 'Lopinavir',
                       'Detection approach', 'DurationRibavirin', 'DurationLopinavir',
                       'DurationOseltamivir', 'DurationArbidol', 'IgM', 'IgG', 'IgGGrade2',
                       'Oxygen therapy approach', 'IgMGrade2', 'Interferon']
    data.drop(columns=delete_features, inplace=True)

    # 2.
    # delete Grade2 删除人为创造的分级特征Grade2
    if delete_grade:
        columns_ = data.columns.values
        for col in columns_:
            if 'Grade2' in col:
                data.drop(columns=[col], inplace=True)

    # 3.
    # delete last 20 features in features rank of rf,gbdt,ls, svm
    path = './feature_select/'+'20200509-15-58-'
    n_last_delete = delete_last_features(path+'rf_feature_importance.xlsx',
                                         path+'gbdt_feature_importance.xlsx',
                                         path+'lr_feature_importance.xlsx',
                                         path+'svm_feature_importance.xlsx',
                                         20,
                                         3)
    data.drop(columns=n_last_delete, inplace=True)

    data.to_excel(filename+'-COVID_FillNan_delete_features.xlsx', index=False)

    index = data.index[np.where(np.isnan(data))[0]].values
    assert len(index) == 0

    ID = data.get(['ID']).values.reshape(len(data), 1)
    data.drop(columns=['ID'], inplace=True)

    columns = data.columns.values[:-1]
    label = data.get(['Death']).values.reshape(len(data), 1)
    data.drop(columns=['Death'], inplace=True)
    data_train = data.values
    return data_train, label, columns, ID


def combine_features_rank(delete_grade=False):
    '''
    对 rf,gbdt, lr, svm的排序结果进行加权平均，
    :param delete_grade:
    :return:
    '''
    model_feature = {}
    model_list = ['rf', 'gbdt', 'lr', 'svm']
    for model in model_list:
        path = './results/'+'20200512-13-00-40-stack-yes/'+model+'_feature_importance.xlsx'
        columns = pd.read_excel(path).columns.values.tolist()
        index = [i+1 for i in range(len(columns))]
        model_feature[model] = dict(zip(columns, index))
    # 每种模型排名对应的权重，用对应的F1值表示
    weights = {'rf': 0.77, 'gbdt': 0.7, 'lr': 0.75, 'svm': 0.75}  # , 'LR': 0.81, 'SVM': 0.74}
    columns = list(model_feature['rf'].keys())
    new_rank = {}
    for col in columns:
        s = 0
        for key in model_list:
            s = s + model_feature[key][col] * weights[key]
        new_rank[col] = float(1.0/float(s/4.0))
    new_rank_df = pd.DataFrame(new_rank, index=[0])

    if delete_grade:
        grade = []
        original = []
        for col in columns:
            if 'Grade2' in col:
                grade.append(col)
            else:
                original.append(col)
        grade_df = new_rank_df.get(grade)
        original_df = new_rank_df.get(original)
        grade_df.sort_values(by=0, axis=1, ascending=False, inplace=True)
        original_df.sort_values(by=0, axis=1, ascending=False, inplace=True)
        grade_df.to_excel('./data/combine_feature_rank_grade_4.xlsx', index=False)
        original_df.to_excel('./data/combine_feature_rank_original_4.xlsx', index=False)

    new_rank_df.sort_values(by=0, axis=1, ascending=False, inplace=True)
    # new_rank_df.drop(columns=['RRGrade2'], inplace=True)
    new_rank_df.to_excel('./results/20200512-13-00-40-stack-yes/13-00-combine_feature_rank.xlsx', index=False)
    new_rank_df.loc[0, new_rank_df.columns.values[:30]].plot(kind='bar')
    plt.show()

    return new_rank


def data_description(filename):
    '''
    分成死亡和出院两种，统计每个特征的缺失情况
    :param filename:
    :return:
    '''
    data = pd.read_excel(filename, dtype={'Name': str, 'Value': float})
    delete_features = ['SOFA score', 'Corticosteroids', 'DurationCorticosteroids',
                       'Intravenous immunoglobin', 'Antibiotics',
                       'carbostyril', 'cephalosporin', 'broad_spectrum',
                       'Vasoactive', 'DurationVasoactive', 'Lac', 'LacGrade2',
                       'Ribavirin', 'Oxygen therapy', 'DurationInterferon',
                       'Oseltamivir', 'Arbidol', 'Lopinavir',
                       'Detection approach', 'DurationRibavirin', 'DurationLopinavir',
                       'DurationOseltamivir', 'DurationArbidol', 'IgM', 'IgG', 'IgGGrade2',
                       'Oxygen therapy approach', 'IgMGrade2', 'Interferon', 'ID', 'Nucleic1']
    data.drop(columns=delete_features, inplace=True)
    columns = data.columns.values
    for col in columns:
        if 'Grade' in col:
            data.drop(columns=[col], inplace=True)
    n_last_delete = delete_last_features('./data/rf_feature_importance.xlsx',
                                         './data/gbdt_feature_importance.xlsx',
                                         './data/lr_feature_importance.xlsx',
                                         './data/svm_feature_importance.xlsx',
                                         20,
                                         3)
    data.drop(columns=n_last_delete, inplace=True)
    data_death = data.loc[data.Death == 2]
    data_live = data.loc[data.Death == 1]
    res_death = len(data_death) - data_death.count().values
    res_live = len(data_live) - data_live.count().values
    v = np.vstack((res_death, res_live))
    columns = data.columns.values.tolist()
    pd_res = pd.DataFrame(data=v.reshape(2, len(columns)), columns=columns,
                          index=['death'+str(len(data_death)), 'live'+str(len(data_live))])
    pd_res.to_excel('./data_description/'+'Nan-statistic-'+filename[7:], index=True)
    print("")


if __name__ == "__main__":
    # combine_features_rank()
    # data, fm = load_data('./data/COVID-19-xy.xlsx', './data/feature-map.json')
    # test_data_xy, test_label_xy, _, id_xy = generate_data('./data/COVID-19-xy.xlsx')
    # data = fill_nan(data, fm, 'xy')
    # data_description('./data/COVID-19-xy.xlsx')
    # data_description('./data/COVID-19-zf.xlsx')
    # data_description('./data/COVID-19-gg.xlsx')
    print("processed finished!")
from sklearn.externals import joblib
from data_preprocess import generate_train_data, generate_data

path = './save_models/20200626-21-50/'

models = [path+i for i in ['rf.pkl', 'gbdt.pkl', 'lrl2.pkl', 'svm.pkl', 'mlp.pkl']]

# load data x_ 数据已包含feature name 为 dataframe格式
X_train, y_train, X_test_zf, y_test_zf, id_zf = generate_train_data('./data_filter30_svdimpute/SF.xlsx',
                                                                    delete_n_last_features=False,
                                                                    over_sample=False)
x_data_gg, test_label_gg, columns, id_gg = generate_data('./data_filter30_svdimpute/OV.xlsx',
                                                         delete_n_last_features=False)
x_data_xy, test_label_xy, _, id_xy = generate_data('./data_filter30_svdimpute/CHWH.xlsx',
                                                   delete_n_last_features=False)
# predict
for x in [X_test_zf, x_data_gg, x_data_xy]:
    for model in models:
        clf = joblib.load(model)
        result = clf.predict(x)

print('finished!')

import sys
sys.path.append("../")
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import xlearn
import DataPrepare
import tpot
import re



pd.set_option('display.max_columns', None)
origin_train_data = pd.read_csv('train.csv')
origin_test_data = pd.read_csv('test.csv')
print(origin_test_data.shape)

print(origin_train_data.info())
print(origin_train_data.head(5))
# print(train_data.describe())
print("*" * 100)


def lr_model(train_feature, train_labels, test_feature, test_labels):
    # LR  结果为0.71 max_iter=1500
    lr_model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1500)
    lr_model.fit(train_feature, train_labels)
    lr_predict_result = lr_model.predict(test_feature)
    # lr_result = roc_auc_score(test_labels, lr_predict_result)
    # print("AUC of LR is " + str(lr_result))
    return lr_predict_result


def tpot_model(train_feature, train_labels, test_feature, test_labels):
    # TPOT 结果为0.743
    tpot_model = tpot.TPOTClassifier(generations=5, population_size=20, verbosity=2)
    tpot_model.fit(train_feature, train_labels)
    tpot_predict_result = tpot_model.predict(test_feature)
    # tpot_result = roc_auc_score(test_labels, tpot_predict_result)
    # print("AUC of tpot is" + str(tpot_result))
    return tpot_predict_result


def GBDT_model(train_feature, train_labels, test_feature, test_labels):
    # GBDT  n_estimators= 120 0.686
    gbdt_model = GradientBoostingClassifier(n_estimators=80, criterion='friedman_mse')
    gbdt_model.fit(train_feature, train_labels)
    gbdt_predict_result = gbdt_model.predict(test_feature)
    # gbdt_result = roc_auc_score(test_labels, gbdt_predict_result)
    # print("AUC of gbdt is" + str(gbdt_result))
    return gbdt_predict_result


def GBDT_LR_model(train_feature, train_labels, test_feature, test_labels):
    # 8, 6, 0.727
    gbdt_model = GradientBoostingClassifier(n_estimators=8, criterion='friedman_mse', max_leaf_nodes=6)
    gbdt_model.fit(train_feature, train_labels)
    print(gbdt_model.max_leaf_nodes)
    one_hot = OneHotEncoder()
    one_hot.fit(gbdt_model.apply(train_feature)[:, :, 0])
    new_feature = one_hot.transform(gbdt_model.apply(train_feature)[:, :, 0])
    print(gbdt_model.apply(train_feature)[:, :, 0])
    print(new_feature.shape)
    new_test_feature = one_hot.transform(gbdt_model.apply(test_feature)[:, :, 0])
    new_feature_df = pd.DataFrame(new_feature.toarray())
    new_test_feature_df = pd.DataFrame(new_test_feature.toarray())
    dict_vectorizer = DictVectorizer(sparse=False)
    new_test_feature_ndarray = dict_vectorizer.fit_transform(new_feature_df.to_dict(orient='record'))
    train_total_feature = np.hstack((train_feature, new_test_feature_ndarray))
    test_new_feature_ndarray = dict_vectorizer.fit_transform(new_test_feature_df.to_dict(orient='record'))
    test_total_feature = np.hstack((test_feature, test_new_feature_ndarray))

    lr_model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1500)
    lr_model.fit(train_total_feature, train_labels)
    lr_predict_result = lr_model.predict(test_total_feature)
    # lr_result = roc_auc_score(test_labels, lr_predict_result)
    # print("GBDT_LR_result" + str(lr_result))
    return lr_predict_result


def fm_model(train_feature, train_labels, test_feature, test_labels):
    # 0.734
    xlearn.write_data_to_xlearn_format(train_feature, train_labels,"../LeavePredict/1.txt")
    xlearn.write_data_to_xlearn_format(test_feature,test_labels,"../LeavePredict/2.txt")
    fm_model = xlearn.create_fm()
    fm_model.setTrain("../LeavePredict/1.txt")
    fm_model.disableLockFree()
    param = {'task': 'binary', 'lr': 0.2, 'lambda': 0.00002, 'metric': 'acc', 'k': 4,'opt':'sgd','epoch':50}
    fm_model.cv(param)
    fm_model.fit(param,"./model.out")
    fm_model.setSign()
    fm_model.setTest("../LeavePredict/2.txt")
    fm_model.predict("./model.out","./result.txt")
    with open("./result.txt",'r') as f:
        data = f.read()
    result_list = re.findall(r'[0-9]', data)
    result = [int(result_list[i]) for i in range(len(result_list))]
    # predict_result = pd.Series(result,dtype='int64')
    # print("AUC of FM is " + str(roc_auc_score(test_labels,predict_result)))
    predict_result = np.array(result)
    return predict_result


def ffm_model(test_labels):
    # 0.767
    ffm_model = xlearn.create_ffm()
    ffm_model.setTrain('ffm_train_data.txt')
    param = {'task':'binary', 'lr':0.2, 'lambda':0.00002,'k':4,'epoch':400,'opt':'adagrad','metric':'acc'}
    ffm_model.cv(param)
    ffm_model.disableLockFree()
    ffm_model.fit(param,'ffm_model.out')
    ffm_model.setSign()
    ffm_model.setTest('ffm_test_data.txt')
    ffm_model.predict('ffm_model.out','ffm_result.txt')
    with open("./ffm_result.txt",'r') as f:
        data = f.read()
    result_list = re.findall(r'[0-9]', data)
    result = [int(result_list[i]) for i in range(len(result_list))]
    predict_result = pd.Series(result,dtype='int64')
    # print("AUC of FFM is " + str(roc_auc_score(test_labels,predict_result)))
    # predict_result = np.array(result)
    return predict_result


def sum_result(*args):
    seprate_result = args
    temp_result_list = list()
    final_result = list()
    for i in range(len(seprate_result[0])):
        temp_result = 0
        for j in range(len(seprate_result)):
            temp_result += seprate_result[j][i]
        temp_result_list.append(temp_result)
        if seprate_result[-1][i] == 1:
            final_result.append(1)
        elif temp_result_list[i] >= 2:
            final_result.append(1)
        else:
            final_result.append(0)
    return final_result


def generate_submit_file(final_result):
    origin_test_data['Attrition'] = final_result
    need_to_submit_list = ['user_id','Attrition']
    submit_result = origin_test_data[need_to_submit_list]
    submit_result.to_csv(path_or_buf='./submission.csv', sep=',', index=False)


def for_set_well_parameter():
    # 0.747   # 0.77  # 0.76
    train_data = origin_train_data[0:1000][:]
    test_data = origin_train_data[1001:][:]
    train_feature, train_labels, test_feature, test_labels = DataPrepare.get_train_feature_and_train_labels(train_data,
                                                                                                            test_data,
                                                                                                            0, 1001)
    DataPrepare.prepare_train_data_for_ffm(train_data, test_data, 0, 1001)
    lr_result = lr_model(train_feature, train_labels, test_feature, test_labels)
    tpot_result = tpot_model(train_feature, train_labels, test_feature, test_labels)
    fm_result = fm_model(train_feature, train_labels, test_feature, test_labels)
    ffm_result = ffm_model(test_labels)
    gbdt_result = GBDT_model(train_feature, train_labels, test_feature, test_labels)
    gbdt_lr_result = GBDT_LR_model(train_feature, train_labels, test_feature, test_labels)

    print(lr_result)
    print(tpot_result)
    print(fm_result)
    print(ffm_result)
    print(gbdt_result)
    print(gbdt_lr_result)
    final_result = sum_result(tpot_result, gbdt_lr_result, fm_result, ffm_result)
    final_result = np.array(final_result)
    print(final_result)
    print("AUC of final result is " + str(roc_auc_score(test_labels, final_result)))


def predict_true_result():
    train_data = origin_train_data
    test_data = origin_test_data
    for_fake_test_label_list = ['No' for i in range(origin_test_data.shape[0])]
    test_data['Attrition'] = np.array(for_fake_test_label_list)
    train_feature, train_labels, test_feature, test_labels = DataPrepare.get_train_feature_and_train_labels(train_data,
                                                                                                            test_data,
                                                                                                            0, 0)
    DataPrepare.prepare_train_data_for_ffm(train_data, test_data, 0, 0)
    lr_result = lr_model(train_feature, train_labels, test_feature, test_labels)
    tpot_result = tpot_model(train_feature, train_labels, test_feature, test_labels)
    fm_result = fm_model(train_feature, train_labels, test_feature, test_labels)
    gbdt_result = GBDT_model(train_feature, train_labels, test_feature, test_labels)
    gbdt_lr_result = GBDT_LR_model(train_feature, train_labels, test_feature, test_labels)
    ffm_result = ffm_model(test_labels)
    print(lr_result)
    print(tpot_result)
    print(fm_result)
    print(gbdt_result)
    print(gbdt_lr_result)
    print(ffm_model(test_labels))
    final_result = sum_result(tpot_result, gbdt_lr_result, fm_result, ffm_result)
    final_result = np.array(final_result)
    print(final_result)
    print(final_result.shape)
    generate_submit_file(final_result)


if __name__ == '__main__':
    # for_set_well_parameter()
    predict_true_result()





import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
import scipy.stats as sts
from sklearn.metrics import precision_score, roc_curve, auc
# import matplotlib.pyplot as plt
# import os
# os.environ["PATH"] += os.pathsep + r'F:\Program Files (x86)\Graphviz2.38\bin'

class MyClassifier:
    def __init__(self):
        self.data_path = 'F:/ModelTest/data/wine.data'
        self.training_data = []
        self.testing_data = []
        self.training_label = []
        self.testing_label = []
        self.label_col_num = -1
        self.thread_num = 3
        self.cv_num = 3 # split fold num
        self.opt_num = 100  # search times for hyper-parameters
        self.best_params = {}
        pass

    def read_data(self):
        data = []
        with open(self.data_path, 'r') as tmp_fi: # read data from local file
            for line in tmp_fi.readlines():
                tmp_row = line.split(',')
                data.append(tmp_row)

        tmp_col_names = ['f%d'%x for x in range(len(data[0]))]  # ** for testing
        data = pd.DataFrame(data, columns=tmp_col_names)
        data.iloc[:, -1] = data.iloc[:, -1].apply(lambda x: x.strip()) # clear up '\n'

        data = data.drop(tmp_col_names[0], axis=1)

        label = data.iloc[:, self.label_col_num]  # which attribute is label
        data = data.drop(data.columns[self.label_col_num], axis=1)

        numeric_col_names = []  # seperate numeric data and categorical data
        categorical_col_names = []
        for tmp_cn in data.columns:
            try:
                data.loc[:, tmp_cn] = data[tmp_cn].astype('float')
                numeric_col_names.append(tmp_cn)
            except Exception as e:
                categorical_col_names.append(tmp_cn)

        data = data[numeric_col_names]

        label = label.astype('int')
        label = label.apply(lambda x: 1 if x ==1 else 0)

        # data = self.replace([-99,-100,-98], np.nan)  # substitute nan back into dataset

        self.training_data, self.testing_data, self.training_label, self.testing_label = train_test_split(data, label, test_size=0.1, random_state=6)  # split data into training and testing sets


    def optimization(self):  # *********** tune hyper parameters
        params = {
            'booster': 'dart',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'nthread': self.thread_num,
        }

        params_space = {
            'max_depth': sts.randint(),
        }

        for tmp_count in range(self.opt_num):


            kf = KFold(n_splits=self.cv_num, shuffle=False, random_state=6)  # k-fold cross validation
            kf.get_n_splits(self.training_data)

            for train_index, val_index in kf.split(self.training_data):
                x_train, x_val = self.training_data.iloc[train_index], self.training_data.iloc[val_index]
                y_train, y_val = self.training_label[train_index], self.training_label[val_index]
                xgb.DMatrix(x_train, label=y_train)

    def training(self, training_data, training_label, validation_data, validation_label, params):
        pos_count = training_label.sum()
        neg_count = training_label.size - pos_count
        params['scale_pos_weight'] = neg_count / float(pos_count)

        training_dmatrix = xgb.DMatrix(training_data, label=training_label, feature_names=training_data.columns)
        validation_dmatrix = xgb.DMatrix(validation_data, label=validation_label)

        model = xgb.train(params, training_dmatrix, num_boost_round=500, verbose_eval=100, early_stopping_rounds=15, evals=[(validation_dmatrix, 'eval'), (training_dmatrix, 'train')])

        # combine original training data and validation data and train again
        training_data = training_data.append(validation_data)
        training_label = training_label.append(validation_label)
        training_dmatrix = xgb.DMatrix(training_data, label=training_label, feature_names=training_data.columns)

        print('best tree num:', model.best_ntree_limit)

        pos_count = training_label.sum()
        neg_count = training_label.size - pos_count
        params['scale_pos_weight'] = neg_count / float(pos_count)
        model = xgb.train(params, training_dmatrix, num_boost_round=model.best_ntree_limit)

        model.save_model('crd_ver1.m')
        return model

    def predicting(self, model, testing_data, testing_label):
        testing_dmatrix = xgb.DMatrix(testing_data, label=testing_label, feature_names=testing_data.columns)
        pro_pred = model.predict(testing_dmatrix, ntree_limit=model.best_ntree_limit).tolist()
        label_pred = list(map(lambda x: 1 if x > 0.5 else 0, pro_pred))  # convert probability to label

        # accessment
        accessment_figures = {}
        accessment_figures['precision'] = precision_score(testing_label, label_pred, average='micro')
        fpr, tpr, tmp_thrs = roc_curve(testing_label, pro_pred, pos_label=1)
        accessment_figures['ks_score'] = np.max(tpr - fpr)
        accessment_figures['auc'] = auc(fpr, tpr)

        return accessment_figures

    def accessment(self):
        pass

if __name__ == '__main__':
    testing_params = {
        'booster': 'dart',
        'nthread':3,
        'learning_rate':0.1,
        'min_child_weight':0.1,
        'max_depth':5,
        'gamma':0.1,  # min loss reduce
        'subsample':0.9, # select samples
        'colsample_bytree': 0.9, # select features
        'max_delta_step':1,  # non-zero when sample imbalance, (when too few samples in a node, G is close to zero, weight would be very large)
        'reg_lambda':1, # L2 regularization
        'reg_alpha': 0.5,  # L1 regularization
        'scale_pos_weight':1,  # weight for sample imbalance
        'objective': 'binary:logistic',
        # 'num_class':3, # multi-class classification
        'seed':6, # random state
        'eval_metric': 'auc',

        # dart params
        'sample_type': 'uniform',
        'normalize_type': 'tree',
        'rate_drop': 0.3,
    }

    mc = MyClassifier()
    mc.read_data()
    new_training_data, validation_data, new_training_label, validation_label = train_test_split(mc.training_data, mc.training_label, test_size=0.3, random_state=6)  # split data into training and testing sets
    model = mc.training(new_training_data, new_training_label, validation_data, validation_label, testing_params)
    access_figure = mc.predicting(model, mc.testing_data, mc.testing_label)

    print('accessment:\n', access_figure)
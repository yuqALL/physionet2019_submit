# -*- coding: utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn.externals import joblib
import evaluate_sepsis_score as scores

def predict_exist_XGBoost(testFeatures, pb):
    clf = joblib.load('XGBoost.pkl')
    # 分类
    label = clf.predict(testFeatures)
    prob = clf.predict_proba(testFeatures)

    # print(label, prob)
    for i in range(len(label)):
        if (prob[i][1] > pb):
            label[i] = 1
        else:
            label[i] = 0

    return (np.array(prob)[:, 1], label)

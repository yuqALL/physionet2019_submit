# -*- coding: UTF-8 -*-
# !/usr/bin/python

import sys
import xgboost as xgb
import numpy as np
from sklearn.externals import joblib
import os

x_mean = np.array([
    83.8996, 97.0520, 36.8055, 126.2240, 86.2907,
    66.2070, 18.7280, 33.7373, -3.1923, 22.5352,
    0.4597, 7.3889, 39.5049, 96.8883, 103.4265,
    22.4952, 87.5214, 7.7210, 106.1982, 1.5961,
    0.6943, 131.5327, 2.0262, 2.0509, 3.5130,
    4.0541, 1.3423, 5.2734, 32.1134, 10.5383,
    38.9974, 10.5585, 286.5404, 198.6777,
    60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551])

x_std = np.array([
    17.6494, 3.0163, 0.6895, 24.2988, 16.6459,
    14.0771, 4.7035, 11.0158, 3.7845, 3.1567,
    6.2684, 0.0710, 9.1087, 3.3971, 430.3638,
    19.0690, 81.7152, 2.3992, 4.9761, 2.0648,
    1.9926, 45.4816, 1.6008, 0.3793, 1.3092,
    0.5844, 2.5511, 20.4142, 6.4362, 2.2302,
    29.8928, 7.0606, 137.3886, 96.8997,
    16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367])


def get_sepsis_score(feature, model):
    feature = genFeature(feature, False)
    # generate predictions
    label = model.predict(feature)
    prob = model.predict_proba(feature)
    # print(label, prob)
    pb = 0.1
    for i in range(len(label)):
        if (prob[i][1] > pb):
            label[i] = 1
        else:
            label[i] = 0
    return np.array(prob)[:, -1], label[-1]


def load_sepsis_model():
    clf = joblib.load("./XGBoost.pkl")
    return clf


# 输入所有数据特征
def genFeature(feature, isTrain=False):
    exlen = 16
    feature = np.array(feature)
    if np.ndim(feature) == 1:
        tNan = np.zeros(exlen)
        tNan[:] = np.nan
        feature = np.hstack((feature, tNan, tNan, tNan, tNan, tNan, feature, tNan, feature, feature, feature))
        return feature
    # feature = data[0][:, :-1]
    h, w = np.array(feature).shape
    for j in range(w):
        for i in range(h):
            if np.isnan(feature[i, j]):
                feature[i, j] = searchNearValue(i, feature[:, j], 3, isTrain)

    for j in range(w):
        for i in range(h):
            if np.isnan(feature[i, j]):
                feature[i, j] = x_mean[j]

    grad1 = featureGrad(feature[:, :exlen], isTrain)

    grad12 = featureGrad_12(feature[:, :exlen], isTrain)

    grad24 = featureGrad_24(feature[:, :exlen], isTrain)

    grad = np.hstack((grad1, grad12, grad24))
    h, w = grad.shape
    for j in range(w):
        for i in range(h):
            if np.isnan(grad[i, j]):
                grad[i, j] = searchNearValue(i, grad[:, j], 3, isTrain)

    mutation = mutationFactor(feature[:, :exlen], isTrain)
    mutationMax = muFactor_max(feature[:, :exlen])

    fSum = featureSum(feature[:, :exlen])
    fSum8 = featureSum_8h(feature[:, :exlen])
    fMax = featureMax(feature[:, :exlen])
    fMin = featureMin(feature[:, :exlen])
    fMean = featureMean(feature[:, :exlen])

    feature = np.hstack((feature, grad, mutation, mutationMax, fSum, fSum8,
                         fMax, fMin, fMean))
    return feature


def searchNearValue(index, list, range, isTrain=False):
    indexL = index
    indexH = index
    while indexL >= max(index - range, 0) and indexH < min(index + range, len(list)):
        if np.isnan(list[indexL]) == False:
            return list[indexL]
        if isTrain:
            if np.isnan(list[indexH]) == False:
                return list[indexH]
            else:
                indexH = indexH + 1
        indexL = indexL - 1
    return list[index]


def featureGrad(feature, isTrain=False):
    h, w = np.array(feature).shape
    tNan = np.zeros(w)
    tNan[:] = np.nan
    grad = np.copy(feature)
    for i in range(h):
        if i > 0:
            grad[i, :] = feature[i, :] - feature[i - 1, :]
        else:
            grad[i, :] = tNan
    if h > 1 and isTrain:
        grad[0, :] = grad[1, :]
    return grad


def featureGrad_12(feature, isTrain=False):
    h, w = np.array(feature).shape
    grad = np.full(np.array(feature).shape, np.nan)
    if h >= 12:
        for i in range(h):
            if i >= 12:
                grad[i, :] = feature[i, :] - feature[i - 12, :]
            elif i >= 6:
                grad[i, :] = feature[i, :] - feature[0, :]
        # if isTrain:
        # grad[0:6, :] = np.full((6, w), grad[6, :])
        # print(grad[0:7, :])
        return grad
    return grad


def featureGrad_24(feature, isTrain=False):
    h, w = np.array(feature).shape
    grad = np.full(np.array(feature).shape, np.nan)
    if h >= 24:
        grad = np.copy(feature)
        for i in range(h):
            if i >= 24:
                grad[i, :] = feature[i, :] - feature[i - 24, :]
            elif i >= 15:
                grad[i, :] = feature[i, :] - feature[0, :]
        # if isTrain:
        # grad[0:15, :] = np.full((15, w), grad[15, :])
        return grad
    return grad


def mFCac(data):
    h, w = data.shape
    m_t = np.nanmean(data, axis=0)
    s_t = np.nanstd(data, axis=0)
    for i in range(w):
        if np.isnan(m_t[i]):
            m_t[i] = x_mean[i]
        if np.isnan(s_t[i]):
            s_t[i] = x_std[i]
        if m_t[i] < 0.001 and m_t[i] > 0:
            m_t[i] = 0.001
        elif m_t[i] > -0.001 and m_t[i] < 0:
            m_t[i] = -0.001
    return np.divide(s_t, m_t)


def mutationFactor(feature, isTrain=False):
    f = np.array(feature)
    h, w = f.shape
    mutation = []
    tNan = np.zeros(w)
    tNan[:] = np.nan
    for i in range(h):
        if i > 0:
            mutation.append(mFCac(f[0:i + 1, :]))
        else:
            mutation.append(tNan.tolist())

    if isTrain:
        mutation = np.array(mutation)
        if h > 1:
            mutation[0, :] = mutation[1, :]
    return mutation


def muFactor_max(feature):
    f = np.array(feature)
    h, w = f.shape
    mutation = []
    tNan = np.zeros(w)
    tNan[:] = np.nan
    validV = 0
    for i in range(h):
        if i < 11:
            mutation.append(tNan.tolist())
        else:
            value = f[i - 11:i + 1, :]
            mV = mFCac(value)
            if validV >= 2:
                splitV = np.nanmean(mutation, axis=0)
                temp = np.zeros_like(splitV)
                for j in range(len(splitV)):
                    if splitV[j] > mV[j]:
                        min = np.nanmin(np.array(mutation)[:, j], axis=0)
                        temp[j] = [min, mV[j]][bool(min > mV[j])]
                    else:
                        max = np.nanmax(np.array(mutation)[:, j], axis=0)
                        temp[j] = [max, mV[j]][bool(max < mV[j])]
                mutation.append(temp)
            else:
                mutation.append(mV)
                validV += 1
    return mutation


def featureSum(feature):
    f = np.array(feature)
    h, w = f.shape
    sum = []
    thred = np.full((h, w), 10000)
    for i in range(h):
        temp = np.min(np.vstack((np.sum(f[0:i + 1, :], axis=0), thred)), axis=0)
        sum.append(temp)
    return sum


def featureSum_8h(feature):
    f = np.array(feature)
    h, w = f.shape
    sum = []
    tNan = np.zeros(w)
    tNan[:] = np.nan
    for i in range(h):
        if i < 7:
            temp = tNan.tolist()
        else:
            temp = np.nansum(f[i - 7:i + 1, :], axis=0)
        sum.append(temp)
    return sum


def featureMax(feature):
    f = np.array(feature)
    h, w = f.shape
    m = []
    for i in range(h):
        temp = np.nanmax(f[:i + 1, :], axis=0)
        m.append(temp)
    return m


def featureMin(feature):
    f = np.array(feature)
    h, w = f.shape
    m = []
    for i in range(h):
        temp = np.nanmin(f[:i + 1, :], axis=0)
        m.append(temp)
    return m


def featureMean(feature):
    f = np.array(feature)
    h, w = f.shape
    m = []
    for i in range(h):
        temp = np.nanmean(f[:i + 1, :], axis=0)
        m.append(temp)
    return m


def featureMedian(feature):
    f = np.array(feature)
    h, w = f.shape
    m = []
    for i in range(h):
        temp = np.nanmedian(f[:i + 1, :], axis=0)
        m.append(temp)
    return m

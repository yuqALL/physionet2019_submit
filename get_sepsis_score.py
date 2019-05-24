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

mFMax = []
fSum_pre = []
fmax = []
fmin = []
fmean = []
grad_temp = []


def get_sepsis_score(feature, model):
    feature = genFeature(feature)
    # generate predictions
    label = model.predict(feature)
    prob = model.predict_proba(feature)
    # print(label, prob)
    pb = 0.1
    # print(prob)
    if (prob[0][1] > pb):
        label = 1
    else:
        label = 0
    return prob[0][1], label


def load_sepsis_model():
    clf = joblib.load("./XGBoost.pkl")
    return clf


# 输入所有数据特征
def genFeature(data):
    global grad_temp
    exlen = 2
    # feature = data[:, :-3]
    feature = data
    h, w = feature.shape

    for j in range(w):
        for i in range(h):
            if np.isnan(feature[i, j]):
                feature[i, j] = searchNearValue(i, feature[:, j], 3, True)

    for j in range(w):
        for i in range(h):
            if np.isnan(feature[i, j]):
                feature[i, j] = x_mean[j]

    grad1 = Grad1(feature[:, :exlen])

    grad12 = Grad12(feature[:, :exlen])

    grad24 = Grad24(feature[:, :exlen])

    grad = np.hstack((grad1, grad12, grad24))

    if h == 1:
        grad_temp = []
        grad_temp.append(grad)
    else:
        grad_temp.append(grad)

    # print(grad)
    temp = np.array(grad_temp)
    h, w = temp.shape
    for j in range(w):
        for i in range(h):
            if np.isnan(temp[i, j]):
                temp[i, j] = searchNearValue(i, temp[:, j], 3, True)

    grad = temp[-1, :]
    grad = np.reshape(grad, (1, w))
    # print("after", grad)
    mutation = mFactor(feature[:, :exlen])
    mutationMax = mFactorMax(feature[:, :exlen])

    fSum = f_sum(feature[:, :exlen])
    fSum8 = f_sum8h(feature[:, :exlen])
    fMax = f_max(feature[:, :exlen])
    fMin = f_min(feature[:, :exlen])
    fMean = f_mean(feature[:, :exlen])

    # print(feature.shape)
    # print(grad.shape)
    # print(mutation.shape)
    # print(mutationMax.shape)
    # print(fSum.shape)
    # print(fSum8.shape)
    # print(fMax.shape)
    # print(fMin.shape)
    # print(fMean.shape)

    f = np.hstack((feature[h - 1:h], grad, mutation, mutationMax, fSum, fSum8,
                   fMax, fMin, fMean))
    return f


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


def Grad1(data):
    h, w = data.shape
    grad = np.zeros(w)
    grad[:] = np.nan
    if h > 1:
        grad = data[-1, :] - data[-2, :]
    return grad


def Grad12(data):
    h, w = data.shape
    grad = np.zeros(w)
    grad[:] = np.nan
    if h >= 13:
        grad = data[-1, :] - data[-12, :]
    elif h >= 7:
        grad = data[-1, :] - data[-7, :]
    return grad


def Grad24(data):
    h, w = data.shape
    grad = np.zeros(w)
    grad[:] = np.nan
    if h >= 25:
        grad = data[-1, :] - data[-25, :]
    elif h >= 16:
        grad = data[-1, :] - data[-16, :]
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


def mFactor(data):
    h, w = data.shape
    mF = np.zeros((1, w))
    mF[0, :] = np.nan
    if h > 1:
        mF[0, :] = mFCac(data)
    return mF


def mFactorMax(data):
    global mFMax
    h, w = data.shape
    mF = np.zeros((1, w))
    mF[:] = np.nan
    if h == 1:
        mFMax = []
    if h > 11:
        mF = mFCac(data[h - 12:h, :])
        if len(mFMax) >= 2:
            print('do here')
            splitV = np.nanmean(np.array(mFMax), axis=0)
            for j in range(len(splitV)):
                if splitV[j] > mF[j]:
                    min = np.nanmin(np.array(mFMax)[:, j], axis=0)
                    mF[j] = [min, mF[j]][bool(min > mF[j])]
                else:
                    max = np.nanmax(np.array(mFMax)[:, j], axis=0)
                    mF[j] = [max, mF[j]][bool(max < mF[j])]
        mFMax.append(mF)
        print(mFMax)
        mF = np.reshape(mF, (1, w))
    return mF


def f_sum(data):
    global fSum_pre
    # print(fSum_pre, data.shape)
    h, w = data.shape
    if h == 1:
        fSum_pre = data
        # print(fSum_pre)
        return data
    else:
        thred = np.full((w), 10000)
        s = np.vstack((fSum_pre, data[-1, :]))
        temp = np.nanmin(np.vstack((np.nansum(s, axis=0), thred)), axis=0)
        fSum_pre = temp
        return np.reshape(temp, (1, w))


def f_sum8h(data):
    h, w = data.shape
    s = np.zeros((1, w))
    s[0, :] = np.nan
    if h >= 8:
        s[0, :] = np.nansum(data[h - 8:h, :], axis=0)
    return s


def f_max(data):
    global fmax
    h, w = data.shape

    if h == 1:
        fmax = data
        return data

    m = np.zeros(w)
    m[:] = np.nan
    m = np.nanmax(np.vstack((fmax, data[-1, :])), axis=0)
    fmax = m
    return np.reshape(m, (1, w))


def f_min(data):
    global fmin
    h, w = data.shape

    if h == 1:
        fmin = data
        return data

    m = np.zeros(w)
    m[:] = np.nan
    m = np.nanmin(np.vstack((fmin, data[-1, :])), axis=0)
    fmin = m
    return np.reshape(m, (1, w))


def f_mean(data):
    global fmean
    h, w = data.shape
    if h == 1:
        fmean = data
        return data

    m = np.zeros(w)
    m[:] = np.nan
    m = np.nanmean(np.vstack((fmean, data[-1, :])), axis=0)
    fmean = m
    return np.reshape(m, (1, w))


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


if __name__ == "__main__":
    # test = GetFeatures.readData('./training/p000050.psv')
    # feature = GetFeatures.getFeature(test)

    test = np.arange(1, 101, 1)
    test = np.reshape(test, (50, 2))
    result = []
    print("src", test)
    for t in range(50):
        current_data = test[:t + 1]
        if t==0:
            result=genFeature(current_data)
        else:
            result=np.vstack((result,genFeature(current_data)))

    print("result", result.shape)

# -*- coding: UTF-8 -*-
# !/usr/bin/python

import numpy as np
import joblib
import os, sys

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
grad_temp = []
hess_temp = []
All_grad1 = []
All_grad12 = []
All_grad24 = []
All_hess1 = []


def get_sepsis_score(feature, model):
    feature = genFeature(feature, True)
    # print(feature.dtype)
    # generate predictions
    label = model.predict(feature)
    prob = model.predict_proba(feature)
    # print(label, prob)
    # pb = 0.1
    # # print(prob)
    # if (prob[0][1] > pb):
    #     label = 1
    # else:
    #     label = 0
    return prob[0][1], label


def load_sepsis_model():
    clf = joblib.load('EasyEnsembleLightGBM.pkl')
    return clf

# 输入所有数据特征
def genFeature(data, isBayes=True):
    global All_grad1, All_grad12, All_grad24, All_hess1
    exlen = 34
    # feature = data[:, :-1]
    feature = data
    h, w = feature.shape
    if h == 1:
        All_grad1 = []
        All_grad12 = []
        All_grad24 = []
        All_hess1 = []

    for j in range(w):
        for i in range(h):
            if np.isnan(feature[i, j]):
                feature[i, j] = searchNearValue(i, feature[:, j], h)

    for j in range(w):
        for i in range(h):
            if np.isnan(feature[i, j]):
                feature[i, j] = x_mean[j]

    # norm = data_norm(feature)
    res = residual_value(feature[:, :exlen])

    rMax = np.nanmax(res[-1])
    rMin = np.nanmin(res[-1])
    rMean = np.nanmean(res[-1])
    rSum = np.nansum(res[-1])
    rStat = np.hstack((rMax, rMin, rMean, rSum))
    rStat = np.reshape(rStat, (1, len(rStat)))

    grad1 = Grad1(res)
    grad12 = Grad12(res)
    grad24 = Grad24(res)
    grad = np.hstack((grad1, grad12, grad24))
    grad = np.reshape(grad, (1, len(grad)))

    gMax = np.nanmax(grad, axis=1)
    gMin = np.nanmin(grad, axis=1)
    gMean = np.nanmean(grad, axis=1)
    gSum = np.nansum(grad, axis=1)
    gStat = np.hstack((gMax, gMin, gMean, gSum))
    gStat = np.reshape(gStat, (1, len(gStat)))

    All_grad1.append(grad[0, :exlen])
    All_grad12.append(grad[0, exlen:2 * exlen])
    All_grad24.append(grad[0, 2 * exlen:3 * exlen])
    hess1 = Grad1(np.array(All_grad1))
    hess12 = Grad12(np.array(All_grad12))
    hess24 = Grad24(np.array(All_grad24))
    hess = np.hstack((hess1, hess12, hess24))
    hess = np.reshape(hess, (1, len(hess)))
    All_hess1.append(hess1)

    hMax = np.nanmax(hess, axis=1)
    hMin = np.nanmin(hess, axis=1)
    hMean = np.nanmean(hess, axis=1)
    hSum = np.nansum(hess, axis=1)
    hStat = np.hstack((hMax, hMin, hMean, hSum))
    hStat = np.reshape(hStat, (1, len(hStat)))

    mutation = mFactor(res)
    mutationMax = mFactorMax(res)
    # mutation12 = mFactor12(res)
    mutation12h = mFactor12h(res)
    mu = np.hstack((mutation, mutationMax, mutation12h))

    mMax = np.nanmax(mu, axis=1)
    mMin = np.nanmin(mu, axis=1)
    mMean = np.nanmean(mu, axis=1)
    mSum = np.nansum(mu, axis=1)
    mStat = np.hstack((mMax, mMin, mMean, mSum))
    mStat = np.reshape(mStat, (1, len(mStat)))
    # print(mStat.shape)

    fSum = f_sum(res)
    fSum8 = f_sum8h(res)
    fMax = f_max(res)
    fMin = f_min(res)
    fMean = f_mean(res)
    fVar = f_var(res)
    stat = np.hstack((fSum, fSum8, fMax, fMin, fMean, fVar))

    # fCov = cov_1d(feature[:, :exlen], [1, 2, 1])
    fCov = covFilter(feature[:, :exlen])
    norm = normalization(feature[:, :exlen])
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    fCov2 = cov_2d(norm, kernel / 16.0)
    laplace_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    fCov_laplace = cov_2d(norm, laplace_kernel)
    cov = np.hstack((fCov, fCov2, fCov_laplace))

    fEnergy = abs_energy(res)
    gAbsSC, hAbsSC = absolute_sum(np.array(All_grad1), np.array(All_hess1))
    rAboveZero = count_above_zero(res)
    rBelowZero = count_below_zero(res)
    freqMax = fre_max(res)
    newf = np.hstack((fEnergy, gAbsSC, hAbsSC, rAboveZero, rBelowZero, freqMax))
    # feature = cov

    # print(grad.shape)
    # print(hess.shape)
    # print(mu.shape)
    # print(stat.shape)
    # print(cov.shape)
    # print(rStat.shape)
    # f = cov
    f = np.hstack((feature[h - 1:h, :], res[h - 1:h, :], grad, hess, mu, stat, cov, rStat, gStat, hStat, mStat, newf))
    f = ImputerMissingValue(f, 40, 'median')
    f = np.array(f).astype('float32')
    return f


def ImputerMissingValue(feature, start=0, way='median'):
    if way == 'median':
        imr = np.load("feature_median_numpy.npy")
    elif way == 'mean':
        imr = np.load("feature_mean_numpy.npy")

    h, w = feature.shape
    for i in range(h):
        for j in range(w - start):
            if np.isnan(feature[i, j + start]):
                feature[i, j + start] = imr[j + start]
    return feature


def searchNearValue(index, list, range):
    indexL = index
    indexH = index
    while indexL >= max(index - range, 0) and indexH < min(index + range, len(list)):
        if np.isnan(list[indexL]) == False:
            return list[indexL]
        indexL = indexL - 1
    return list[index]


fMax = np.array([
    280.00, 100.00, 50.00, 300.00, 300.00, 300.00, 100.00, 100.00,
    100.00, 55.00, 4000.00, 7.93, 100.00, 100.00, 9961.00, 268.00,
    3833.00, 27.90, 145.00, 46.60, 37.50, 988.00, 31.00, 9.80,
    18.80, 27.50, 49.60, 440.00, 71.70, 32.00, 250.00, 440.00,
    1760.00, 2322.00, 100.00, 1.00, 1.00, 1.00, 23.99, 336.00
])

fMin = np.array([
    20.00, 20.00, 20.90, 20.00, 20.00, 20.00, 1.00, 10.00,
    -32.00, 0.00, -50.00, 6.62, 10.00, 23.00, 3.00, 1.00,
    7.00, 1.00, 26.00, 0.10, 0.01, 10.00, 0.20, 0.20,
    0.20, 1.00, 0.10, 0.01, 5.50, 2.20, 12.50, 0.10,
    34.00, 1.00, 14.00, 0.00, 0.00, 0.00, -5366.86, 1.00
])

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

fN = fMax - fMin


def normalization(item):
    h, w = item.shape
    global fMin, fMax, fN
    temp = (item - fMin[:w]) / fN[:w]
    norm = np.ones_like(temp)
    lower = np.zeros_like(temp)
    temp = np.minimum(temp, norm)
    result = np.maximum(temp, lower)
    return result


def data_norm(data):
    norm = (data[-1] - x_mean) / x_std
    return np.reshape(norm, (1, len(norm)))


def Grad1(data):
    h, w = data.shape
    grad = np.zeros(w)
    grad[:] = np.nan
    if h > 1:
        grad = data[-1, :] - data[-2, :]
    return grad


def covFilter(feature):
    h, w = feature.shape
    f = np.full((h + 2, w), 0)
    f[2:, :] = feature
    result = np.full((1, w), np.nan)

    result[0, :] = (f[-3, :] + f[-2, :] * 2 + f[-1, :]) / 4
    return np.reshape(result, (1, w))


def Grad12(data):
    h, w = data.shape
    grad = np.zeros(w)
    grad[:] = np.nan
    if h >= 13:
        grad = data[-1, :] - data[-13, :]
    elif h >= 7:
        grad = data[-1, :] - data[0, :]

    return grad


def Grad24(data):
    h, w = data.shape
    grad = np.zeros(w)
    grad[:] = np.nan
    if h >= 25:
        grad = data[-1, :] - data[-25, :]
    elif h >= 16:
        grad = data[-1, :] - data[0, :]
    return grad


def mFCac(data):
    h, w = data.shape
    m_t = np.nanmean(data, axis=0)
    s_t = np.nanstd(data, axis=0)
    for i in range(w):
        if np.isnan(m_t[i]):
            # m_t[i] = x_mean[i]
            m_t[i] = 0.001
        if np.isnan(s_t[i]):
            s_t[i] = x_std[i]
        if m_t[i] < 0.001 and m_t[i] >= 0:
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
            # splitV = np.nanmean(np.array(mFMax), axis=0)
            splitV = np.nanmedian(np.array(mFMax), axis=0)
            for j in range(len(splitV)):
                if splitV[j] > mF[j]:
                    min = np.nanmin(np.array(mFMax)[:, j], axis=0)
                    mF[j] = [min, mF[j]][bool(min > mF[j])]
                else:
                    max = np.nanmax(np.array(mFMax)[:, j], axis=0)
                    mF[j] = [max, mF[j]][bool(max < mF[j])]
        mFMax.append(mF)
        # print(mFMax)
        mF = np.reshape(mF, (1, w))
    return mF


index_i = 0
m_max = 0


def mFactor12(data):
    h, w = data.shape
    global index_i, m_max
    if h == 1:
        index_i = 0
        m_max = 0
    mF = np.zeros((1, w))
    mF[0, :] = np.nan

    if h < 12:
        return mF
    elif h < 37:
        for j in range(h - 11):
            m_t_mean = np.nanmean(data[j:j + 11, :], axis=0)
            m_t_max = np.nanmax(data[j:j + 11, :], axis=0)
            m_t_min = np.nanmin(data[j:j + 11, :], axis=0)
            m_t = m_t_mean / ((m_t_max - m_t_min) + 1e-3)
            m_t_all = np.nansum(m_t[:8])
            if m_t_all > m_max:
                m_max = m_t_all
                index_i = j
        if h - index_i <= 12:
            mF[0, :] = mFCac(data[index_i:h, :])
        else:
            k = int((h - index_i - 1) / 12)
            # print('h', h, index_i + k * 12, index_i)
            mF[0, :] = mFCac(data[index_i + k * 12:h, :])
    else:
        k = int((h - index_i - 1) / 12)
        # print('h', h, index_i + k * 12, index_i)
        mF[0, :] = mFCac(data[index_i + k * 12:h, :])
    return mF


def mFactor12h(data):
    h, w = data.shape
    mF = np.zeros((1, w))
    mF[0, :] = np.nan
    if h >= 12:
        mF[0, :] = mFCac(data[h - 12:h, :])
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
        thred = np.full((w), 1000000)
        s = np.vstack((fSum_pre, data[-1, :]))
        temp = np.nanmin(np.vstack((np.nansum(s, axis=0), thred)), axis=0)
        fSum_pre = temp
        return np.reshape(temp, (1, w))


def featureSum(feature):
    f = np.array(feature)
    h, w = f.shape
    sum = []
    thred = np.full((h, w), 1000000)
    for i in range(h):
        temp = np.min(np.vstack((np.sum(f[0:i + 1, :], axis=0), thred)), axis=0)
        sum.append(temp)
    return sum


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
    h, w = data.shape
    m = np.zeros(w)
    m[:] = np.nan
    m = np.nanmean(data, axis=0)
    return np.reshape(m, (1, w))


def f_var(data):
    h, w = data.shape
    m = np.zeros(w)
    m[:] = np.nan
    m = np.nanvar(data, axis=0)
    return np.reshape(m, (1, w))


def f_median(data):
    h, w = data.shape
    m = np.zeros(w)
    m[:] = np.nan
    m = np.nanmedian(data, axis=0)
    return np.reshape(m, (1, w))


def residual_value(feature):
    h, w = feature.shape
    data = np.full((h, w), x_mean[0:w])
    return feature - data


def cov_1d(feature, kernel):
    n = len(kernel)
    h, w = feature.shape
    result = np.full((1, w), np.nan)
    k_sum = 0
    if h < n:
        for i in range(h):
            result += kernel[n - i - 1] * feature[-i - 1]
            k_sum += kernel[n - i - 1]
        result = result / k_sum
        return np.reshape(result, (1, w))
    f = feature[-n:, :]
    for i in range(n):
        result += kernel[i] * f[-n + i, :]
    result = result / np.sum(kernel)
    return np.reshape(result, (1, w))


def cov_2d(feature, kernel):
    h, w = feature.shape
    kh, kw = kernel.shape
    if kw % 2 == 0:
        print("kernel error")
    tmp = np.zeros((h + kh - 1, w + (kw - 1)), dtype='float32')
    tmp[kh - 1:, kw - 1:] = feature
    result = np.full((1, w), np.nan)
    for j in range(w):
        t = tmp[-kh:, j:j + kw] * kernel
        result[0, j] = np.sum(t)
    return result


def compare(value, left, right):
    if np.isnan(value):
        return 0
    if value > right:
        return 1
    elif value > left:
        return 0
    else:
        return -1


def gen_obs(list):
    result = []
    result.append(compare(list[0], 60, 150))
    result.append(compare(list[1], 91, 99))
    result.append(compare(list[2], 36.5, 37.3))
    result.append(compare(list[3], 90, 139))
    result.append(compare(list[4], 65, 110))
    result.append(compare(list[5], 50, 92))
    result.append(compare(list[6], 12, 60))
    result.append(compare(list[7], 35, 45))
    result.append(compare(list[8], -10, 6))
    result.append(compare(list[9], 23, 30))
    result.append(compare(list[10], 0.6, 0.98))
    result.append(compare(list[11], 6, 8))
    return result


def get_obsfeature(feature):
    h, w = feature.shape
    result = []
    for i in range(h):
        result.append(gen_obs(feature[i, :12]))
    return np.reshape(result, (h, 12))


def fre_max(feature):
    f = np.copy(feature)
    h, w = f.shape
    m = np.zeros((1, w))
    if h > 1:
        sp = np.fft.fft(f, axis=0)
        max_f = np.zeros(w)
        for k in range(h):
            max_f = np.max(np.vstack((np.sqrt(sp[k].real ** 2 + sp[k].imag ** 2), max_f)), axis=0)
        m = np.reshape(max_f, (1, w))
    return np.array(m)


energy_pre = []


def abs_energy(feature):
    global energy_pre
    f = np.copy(feature)
    h, w = f.shape
    if h == 1:
        energy_pre = np.zeros(w)
    energy_pre += np.power(feature[-1], 2)
    m = np.reshape(np.copy(energy_pre), (1, w))
    # print(m)
    return m


abs_grad = []
abs_hess = []


def absolute_sum(grad, hess):
    global abs_grad, abs_hess
    ##NaN处理
    h, w = np.array(grad).shape
    gd = np.copy(grad)
    hs = np.copy(hess)
    for j in range(w):
        if np.isnan(gd[-1, j]):
            gd[-1, j] = 0
        if np.isnan(hs[-1, j]):
            hs[-1, j] = 0
    if h == 1:
        abs_grad = np.zeros(w)
        abs_hess = np.zeros(w)
    abs_grad += np.abs(gd[-1])
    abs_hess += np.abs(hs[-1])
    g1 = np.reshape(np.copy(abs_grad), (1, w))
    h1 = np.reshape(np.copy(abs_hess), (1, w))
    return g1, h1


count_above = []


def count_above_zero(feature):
    global count_above
    h, w = np.array(feature).shape
    if h == 1:
        count_above = np.zeros(w)
    count_above += feature[-1] > 0
    m = np.reshape(np.copy(count_above), (1, w))
    return m


count_below = []


def count_below_zero(feature):
    global count_below
    h, w = np.array(feature).shape
    if h == 1:
        count_below = np.zeros(w)
    count_below += feature[-1] < 0
    m = np.reshape(np.copy(count_below), (1, w))
    return m


if __name__ == "__main__":
   
    # feature = GetFeatures.getFeature(test)
    load_sepsis_model()

    # test = np.arange(1, 101, 1)
    # test = np.reshape(test, (50, 2))
    # result = []
    # print("src", test)
    # for t in range(50):
    #     current_data = test[:t + 1]
    #     if t == 0:
    #         result = genFeature(current_data)
    #     else:
    #         result = np.vstack((result, genFeature(current_data)))
    #
    # print("result", result.shape)

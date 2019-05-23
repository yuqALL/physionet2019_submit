# -*- coding: utf-8 -*-
import glob
import numpy as np
import extractFeatures as ef

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


def read_feature_name(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
    return column_names


def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        values = np.loadtxt(f, delimiter='|')
    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        values = values[:, :-1]
    return (values, column_names)


featureName = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH',
               'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
               'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
               'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
               'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']


# 读取结果
def readResult(fileName):
    with open(fileName, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        values = np.loadtxt(f, delimiter='|')
    if column_names[-1] == 'SepsisLabel':
        return values[-1]
    return -1


def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        values = np.loadtxt(f, delimiter='|')
    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        values = values[:, :-1]
    return (values, column_names)


# 读取数据记录文件
def readData(filename):
    with open(filename, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        values = np.loadtxt(f, delimiter='|')

    return (values, column_names)


# 得到训练数据分类结果
def getResult(data):
    return data[0][:, -1]


# 输入所有数据特征
def getFeature(feature, isTrain=False):
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

    exlen = 16
    grad1 = ef.featureGrad(feature[:, :exlen], isTrain)

    grad12 = ef.featureGrad_12(feature[:, :exlen], isTrain)

    grad24 = ef.featureGrad_24(feature[:, :exlen], isTrain)

    grad = np.hstack((grad1, grad12, grad24))
    h, w = grad.shape
    for j in range(w):
        for i in range(h):
            if np.isnan(grad[i, j]):
                grad[i, j] = searchNearValue(i, grad[:, j], 3, isTrain)

    mutation = ef.mutationFactor(feature[:, :exlen], isTrain)
    mutationMax = ef.muFactor_max(feature[:, :exlen])

    fSum = ef.featureSum(feature[:, :exlen])
    fSum8 = ef.featureSum_8h(feature[:, :exlen])
    fMax = ef.featureMax(feature[:, :exlen])
    fMin = ef.featureMin(feature[:, :exlen])
    fMean = ef.featureMean(feature[:, :exlen])
    # print(np.array(fSum).shape)
    # print(np.array(fSum8).shape)
    # print(np.array(fMax).shape)
    # print(np.array(fMin).shape)
    # print(np.array(fMean).shape)
    feature = np.hstack((feature, grad, mutation, mutationMax, fSum, fSum8,
                         fMax, fMin, fMean))
    # print(np.array(feature).shape)
    return feature


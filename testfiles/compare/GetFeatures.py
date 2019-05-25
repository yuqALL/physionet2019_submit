# -*- coding: utf-8 -*-
import glob
import numpy as np
import extractFeatures as ef

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


def genOriginalDataImg(data):
    feature = data[0][:, :-1]
    h, w = np.array(feature).shape
    for j in range(w):
        for i in range(h):
            if np.isnan(feature[i, j]):
                feature[i, j] = x_mean[j]

    return


# 得到训练数据分类结果
def getResult(data):
    return data[0][:, -1]


# 输入所有数据特征
def getOriginFeature(data):
    feature = data[0][:, :-1]
    return feature


# 输入所有数据特征
def getFeature(data, isTrain=False):
    feature = data[:, :-6]

    h, w = feature.shape

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
    # print(np.array(mutation).shape)
    # print(np.array(mutationMax).shape)
    # print(np.array(grad).shape)
    # print(np.array(feature).shape)
    # print(np.array(feature).shape)
    # mutation12 = ef.mutationFactor_12_train(feature[:, :exlen])
    # mutation12h = ef.mutationFactor_12h(feature[:, :exlen])
    # mutation_rotate = ef.mutationFactor_12_rotate(feature)

    # print(np.array(mutation12).shape)
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


def deleInvalidData(data):
    dele = []
    newData = []
    feature = data[0][:, :-1]
    h, w = np.array(feature).shape
    for i in range(h):
        nan = np.isnan(feature[i, :34])
        if np.sum(nan) >= 34:
            # print(feature[i, :34])
            dele.append(i)

    temp = np.delete(data[0], dele, axis=0)
    newData.append(temp)
    newData.append(data[1])
    return newData


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


# 读取训练集特征值
# 输入：特征值文件夹路径、分类结果文件夹路径、非连续特征值名称、连续特征值名称
# 输出：患病特征值、正常特征值、所有特征值、所有特征值标签
def getFeatures2(dataPath):
    filenames = glob.glob(dataPath)
    PosFtr = []
    NegFtr = []
    PosFtrResult = []
    NegFtrResult = []
    AllFtr = []
    AllResult = []
    AllLabel = []
    b = 0
    c = 0

    for index in range(len(filenames)):
        data = readData(filenames[index])
        data = deleInvalidData(data)
        # print(data[0].shape)
        AllLabel.append(data[1])
        feature = getFeature(data)

        # print(np.array(feature).shape)
        labels = getResult(data)
        num_rows = len(labels)
        if np.any(labels):
            is_septic = True
            t_sepsis_optimal = np.argmax(labels)
            labels[max(0, t_sepsis_optimal - 6): min(t_sepsis_optimal + 9 + 1, num_rows)] = 1
        else:
            is_septic = False

        for i in range(num_rows):
            f = feature[i]
            l = labels[i]
            AllFtr.append(f)
            AllResult.append(l)
            if l > 0:
                b += 1
                PosFtr.append(f)
                PosFtrResult.append(l)
            else:
                c += 1
                # print(sesis[i], 'add other', b)
                NegFtr.append(f)
                NegFtrResult.append(l)

    print(np.array(AllFtr).shape)

    print('b', b)
    print('c', c)
    print(len(PosFtrResult))
    print(len(NegFtrResult))
    # for i in range(len(PosFtrResult)):
    #     ran = np.random.randint(0, 39, (5, 8), dtype='uint8')
    #     fe = np.copy(PosFtr[i])
    #     for j in range(5):
    #         temp = np.copy(fe)
    #         for l in range(8):
    #             k = np.random.randint(-3, 4)
    #             temp[ran[j, l]] = fe[ran[j, l]] + k * 0.1 * x_std[ran[j, l]]
    #         PosFtr.append(temp)
    #         PosFtrResult.append(PosFtrResult[i])

    print(len(PosFtrResult))
    print(len(NegFtrResult))

    FtrLabel = [1] * len(PosFtr) + [0] * len(NegFtr)
    print(len(AllFtr))
    print(np.array(AllFtr).shape)
    print(len(AllResult))

    return PosFtr, NegFtr, FtrLabel, AllFtr, AllResult, featureName


# 读取训练集特征值
# 输入：特征值文件夹路径、分类结果文件夹路径、非连续特征值名称、连续特征值名称
# 输出：患病特征值、正常特征值、所有特征值、所有特征值标签
def getFeatures(path, isTrain=False):
    input_files = []
    for i in range(len(path)):
        temp = os.listdir(path[i])
        temp = sorted(f for f in temp if os.path.isfile(os.path.join(path[i], f)) and f.endswith('.psv'))
        for j in range(len(temp)):
            input_files.append(os.path.join(path[i], temp[j]))

    n = len(input_files)

    PosFtr = []
    NegFtr = []
    PosFtrResult = []
    NegFtrResult = []
    AllFtr = []
    AllResult = []
    AllLabel = []
    b = 0
    c = 0

    for index in range(n):
        data = readData(input_files[index])
        # print(data[0].shape)
        data = deleInvalidData(data)
        # print(data[0].shape)
        AllLabel.append(data[1])
        h, w = data[0].shape
        if h < 2:
            continue
            print("shape", data[0].shape)
        feature = getFeature(data, isTrain)

        # print(np.array(feature).shape)
        labels = getResult(data)
        num_rows = len(labels)
        AllFtr.append(feature)
        AllResult.append(labels)

        if np.any(labels):
            is_septic = True
            t_sepsis_optimal = np.argmax(labels)
            labels[max(0, t_sepsis_optimal - 6): min(t_sepsis_optimal + 9 + 1, num_rows)] = 1
            PosFtr.append(feature)
            PosFtrResult.append(labels)
            b += 1
        else:
            is_septic = False
            NegFtr.append(feature)
            NegFtrResult.append(labels)
            c += 1

    print(np.array(AllFtr).shape)

    print('b', b)
    print('c', c)
    print(len(PosFtrResult))
    print(len(NegFtrResult))

    print(len(AllFtr))
    print(np.array(AllFtr).shape)
    print(len(AllResult))

    return PosFtr, PosFtrResult, NegFtr, NegFtrResult, AllFtr, AllResult


# 读取训练集特征值
# 输入：特征值文件夹路径、分类结果文件夹路径、非连续特征值名称、连续特征值名称
# 输出：患病特征值、正常特征值、所有特征值、所有特征值标签
def getOriginFeatures(dataPath):
    filenames = glob.glob(dataPath)
    #    PosRepl1, PosRepl2, NegRepl1, NegRepl2 = GetReplValue.getTrainReplacementValue(dataPath, outcomesPath, featureName1, featureName2)
    #     Threshold1, Threshold2 = GetOutlierThreshold.getOutlierThreshold(dataPath, outcomesPath, featureName1, featureName2)
    PosFtr = []
    NegFtr = []
    PosFtrResult = []
    NegFtrResult = []

    AllFtr = []
    AllResult = []
    AllLabel = []
    b = 0
    c = 0

    for index in range(len(filenames)):
        data = readData(filenames[index])
        # print(data[0].shape)
        AllLabel.append(data[1])
        feature = getOriginFeature(data)

        # print(np.array(feature).shape)
        sesis = getResult(data)
        for i in range(len(sesis)):
            if sesis[i] > 0:
                # if (i + 1) < len(sesis) and sesis[i + 1] < 1:
                #     print(filenames[index])
                t = i - 8
                j = i
                # print(filenames[index])
                # print('before', sesis)
                while (j > 0) and (j > t):
                    j -= 1
                    sesis[j] = 1
                # print('after', sesis)
            break

        for i in range(len(sesis)):
            if sesis[i] > 0:
                b += 1
                PosFtr.append(feature[i])
                PosFtrResult.append(sesis[i])
            else:
                c += 1
                # print(sesis[i], 'add other', b)
                NegFtr.append(feature[i])
                NegFtrResult.append(sesis[i])

    print(np.array(AllFtr).shape)

    print('b', b)
    print('c', c)
    print(len(PosFtrResult))
    print(len(NegFtrResult))
    for i in range(len(PosFtrResult)):
        ran = np.random.randint(0, 39, (5, 8), dtype='uint8')
        fe = np.copy(PosFtr[i])
        for j in range(5):
            temp = np.copy(fe)
            for l in range(8):
                k = np.random.randint(-3, 4)
                temp[ran[j, l]] = fe[ran[j, l]] + k * 0.1 * x_std[ran[j, l]]
            PosFtr.append(temp)
            PosFtrResult.append(PosFtrResult[i])

    AllFtr = np.vstack((PosFtr, NegFtr))
    AllResult = np.hstack((PosFtrResult, NegFtrResult))

    print(len(PosFtrResult))
    print(len(NegFtrResult))

    FtrLabel = [1] * len(PosFtr) + [0] * len(NegFtr)
    print(len(AllFtr))
    print(np.array(AllFtr).shape)
    print(len(AllResult))

    return PosFtr, NegFtr, FtrLabel, AllFtr, AllResult, featureName


if __name__ == "__main__":
    # test = GetFeatures.readData('./training/p000050.psv')
    # feature = GetFeatures.getFeature(test)

    test = np.arange(1, 101, 1)
    test = np.reshape(test, (50, 2))
    print(test)
    result = getFeature(test, True)
    print(result)

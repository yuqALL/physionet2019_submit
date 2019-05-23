import numpy as np

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


def mutationFactor_12_test(feature):
    f = np.array(feature)
    h, w = f.shape
    index_i = 0
    mutation = []
    tNan = np.zeros(w)
    tNan[:] = np.nan
    m_max = 0
    for i in range(h):
        if i < 12:
            mutation.append(tNan.tolist())
        elif i < 24:
            for j in range(i + 1 - 12):
                m_t = np.nanmean(f[j:j + 12, :], axis=0)
                m_t_all = np.nansum(m_t)
                if m_t_all > m_max:
                    m_max = m_t_all
                    index_i = j
            if i - index_i <= 12:
                mutation.append(mFCac(f[index_i:i + 1, :]))
            else:
                k = int((i - index_i) / 12)
                mutation.append(mFCac(f[index_i + k * 12:i + 1, :]))
        else:
            k = int((i - index_i) / 12)
            mutation.append(mFCac(f[index_i + k * 12:i + 1, :]))
    return mutation


# 12小时更新一次
def mutationFactor_12h(feature):
    f = np.array(feature)
    h, w = f.shape
    mutation = []
    tNan = np.zeros(w)
    tNan[:] = np.nan
    for i in range(h):
        if i < 12:
            mutation.append(tNan.tolist())
        else:
            mutation.append(mFCac(f[i - 12:i, :]))
    return mutation


def mutationFactor_12_train(feature):
    f = np.array(feature)
    h, w = f.shape
    m_max = 0
    index_i = 0
    mutation = []
    for i in range(h):
        # m_t = np.mean(f[i:i + 11, :])
        if i + 11 < h:
            m_t = np.mean(f[i:i + 12, :], axis=0)
            m_t_all = np.sum(m_t)
            if m_t_all > m_max:
                m_max = m_t_all
                index_i = i
        if i > 23:
            break

    t = int((h - index_i) / 12) + 2
    for i in range(t):

        if i == 0:
            if index_i > 0:
                temp = mFCac(f[0:index_i, :])
                mutation = np.full((index_i, w), temp)
            else:
                continue
            # print(i, np.array(mutation).shape)
        else:
            j = i - 1
            if (index_i + j * 12 + 12) < h:
                temp = mFCac(f[index_i + j * 12:index_i + i * 12, :])

                if index_i == 0 and i == 1:
                    mutation = np.full((12, w), temp)
                else:
                    mutation = np.vstack((mutation, np.full((12, w), temp)))

                # print(i, np.array(mutation).shape)
            else:
                if (h - 1) >= (index_i + j * 12):
                    temp = mFCac(f[index_i + j * 12:h, :])
                    # print(mutation.shape, temp.shape)
                    if len(mutation):
                        mutation = np.vstack((mutation, np.full((h - index_i - j * 12, w), temp)))
                    else:
                        mutation = np.full((h - index_i - j * 12, w), temp)
                else:
                    continue
    return mutation


def mutationFactor_12_rotate(feature):
    f = np.array(feature)
    # print("f", np.array(f).shape)
    h, w = f.shape
    m_max = 0
    index_i = 0
    mutation = []
    for i in range(h):
        if i == 0:
            m_t = (np.copy(f[0, :]))
            s_t = (np.zeros((8,)))
            mutation.append(np.divide(s_t, m_t))
        elif i < 12:
            m_t = np.mean(f[0:i, :], axis=0)
            s_t = np.std(f[0:i, :], axis=0)
            mutation.append(np.divide(s_t, m_t))
        elif i < 24:
            for j in range(i - 11):
                # m_t = np.mean(f[i:i + 11, :])
                m_t = np.mean(f[j:j + 11, :], axis=0)
                m_t_all = np.sum(m_t)
                if m_t_all > m_max:
                    m_max = m_t_all
                    index_i = j

            if i - index_i <= 12:
                m_t = np.mean(f[index_i:i, :], axis=0)
                s_t = np.std(f[index_i:i, :], axis=0)
                mutation.append(np.divide(s_t, m_t))
            else:
                k = (int((i - index_i) / 12) + 1) % 2
                if k == 0:
                    m_t = np.mean(f[index_i:index_i + 11, :], axis=0)
                    s_t = np.std(f[index_i:index_i + 11, :], axis=0)
                    mutation.append(np.divide(s_t, m_t))
                else:
                    m_t = np.mean(f[index_i + k * 12:i, :], axis=0)
                    s_t = np.std(f[index_i + k * 12:i, :], axis=0)
                    mutation.append(np.divide(s_t, m_t))

        else:
            k = (int((i - index_i) / 12) + 1) % 2
            if k == 0:
                m_t = np.mean(f[index_i:index_i + 11, :], axis=0)
                s_t = np.std(f[index_i:index_i + 11, :], axis=0)
                mutation.append(np.divide(s_t, m_t))
            else:
                if i - index_i > 23:
                    m_t = np.mean(f[index_i + 12:index_i + 23, :], axis=0)
                    s_t = np.std(f[index_i + 12:index_i + 23, :], axis=0)
                    mutation.append(np.divide(s_t, m_t))
                else:
                    m_t = np.mean(f[index_i + 12:i, :], axis=0)
                    s_t = np.std(f[index_i + 12:i, :], axis=0)
                    mutation.append(np.divide(s_t, m_t))

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


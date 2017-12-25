import math
import numpy as np


def simplexNorm(array):
    if isinstance(array, np.ndarray):
        shape = array.shape
        aSum = 0
        if len(shape) == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    aSum += array[i, j]
            for i in range(shape[0]):
                for j in range(shape[1]):
                    array[i, j] = array[i, j] / aSum
        elif len(shape) == 1:
            for i in range(shape[0]):
                aSum += array[i]
            for i in range(shape[0]):
                array[i] /= aSum
    elif isinstance(array, list) or isinstance(array, tuple):
        aSum = 0
        shape = len(array)
        for i in range(shape[0]):
            aSum += array[i]
        for i in range(shape[0]):
            array[i] /= aSum
    else:
        raise ValueError("value type error")


def correlatedMatrixGenerate(shape, scale):
    """
    协相关矩阵生成
    :param shape:
    :param scale:
    :return:
    """
    if len(shape) != 2:
        raise ValueError("shape value error")
    res = np.random.exponential(scale, shape)
    for i in range(shape[0]):
        res[i, i] = 1
    return res


def betaGenerate(shape):
    """
    beta生成
    :param shape: [topicNum,dictNum]
    :return:
    """
    res = np.ndarray(shape, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            res[i, j] = 1 / shape[1]
    return res


def invMatrix(m):
    result = np.matrix(m).getI()
    return result

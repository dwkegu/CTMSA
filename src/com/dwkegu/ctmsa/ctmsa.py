import os
import numpy
import math
import sys
from src.com.dwkegu.ctmsa.model import CTMSAModel
from src.com.dwkegu.ctmsa.dataset import *


def ctmsa():
    args = sys.argv
    wordMap = WordMap()
    model = CTMSAModel(wordMap,100)
    #todo 初始化模型参数
    model.initParameters()
    nytimesDataset = getNYTimesDataset()
    model.setTrainDataset(nytimesDataset.getTrainDocs())
    model.setTestDataset(nytimesDataset.getTestDocs())
    model.inference()




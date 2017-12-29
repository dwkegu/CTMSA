import sys

from CTMSA.src.dataset import *
from CTMSA.src.model import CTMSAModel


def ctmsa():
    args = sys.argv
    wordMap = WordMap()
    model = CTMSAModel(wordMap,100)
    nytimesDataset = getNYTimesDataset()
    model.setTrainDataset(nytimesDataset.getTrainDocs())
    model.setTestDataset(nytimesDataset.getTestDocs())
    model.inference()




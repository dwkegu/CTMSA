import os
import numpy
import csv


NYTimesRawFile = 'f:/tmp/dataset/nytimes/nytimes.csv'

def getYelpDataset():
    pass


def getNYTimesDataset():
    # todo read nytimes dataset
    docsCount = 0
    with open(NYTimesRawFile,'r') as f:
        f_csv = csv.reader(f)
        for line in f_csv:
            docsCount += 1
            # ...

def getWikiDataset():
    pass


class Dataset:
    def __init__(self):
        self._trainDocs = list()
        self._testDocs = list()
        self.docsNum = 0
        self.wordCount = 0
        self.wordMap = None

    def getTrainDocs(self):
        return self._trainDocs

    def getTestDocs(self):
        return self._testDocs
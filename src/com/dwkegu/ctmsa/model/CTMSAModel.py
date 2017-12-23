import os
import sys
import time
import numpy as np
from CTMSA.src.com.dwkegu.ctmsa.util import mathUtil
from CTMSA.src.com.dwkegu.ctmsa.model import Document
from CTMSA.src.com.dwkegu.ctmsa import config


class CTMSAModel:
    """
    correlated topic model with sentence attention
    """

    def __init__(self, wordMap, topicNum=None, paraLambda=None):
        """
        init model
        :param wordMap:
        :param topicNum:
        :param paraLambda:
        """
        self.wordMap = wordMap
        self.K = (config.K if topicNum is None else topicNum)
        self.docs = list()
        self.mu = None
        self.sigma = None
        self.invSigma = None
        self.paraLambda = (config.paraLambda if paraLambda is None else paraLambda)
        self.logoBeta = None
        self.lHood = 0
        self.trainDocs = None
        self.testDocs = None
        # todo init zeta
        self.zeta = 1
        self.oldZeta = 1

    def initParameters(self):
        self.mu = np.random.uniform(0, 1.0, [self.K])
        mathUtil.simplexNorm(self.mu)
        self.sigma = mathUtil.correlatedMatrixGenerate([self.K, self.K], 4.0)
        self.invSigma = mathUtil.invMatrix(self.sigma)
        self.beta = mathUtil.betaGenerate([self.K, self.wordMap.size()])

    def inference(self):
        for doc in self.docs:
            for sentence in doc.sentences:
                self.updateZeta(sentence)
                self.updateGamma(sentence)
                for word in sentence.words:
                    pass

    def updateZeta(self, sentence):
        pass

    def updateGamma(self, sentence):
        pass

    def estimate(self):
        pass

    def setTrainDataset(self, dataset):
        self.trainDocs = dataset

    def setTestDataset(self, dataset):
        self.testDocs = dataset

import os
import sys
import time
import math
import numpy as np
from src.com.dwkegu.ctmsa.util import mathUtil
from src.com.dwkegu.ctmsa.model import Document
from src.com.dwkegu.ctmsa import config
from ..util.FileSaver import FileSaver


class CTMSAModel:
    """
    correlated topic model with sentence attention
    """

    def __init__(self, wordMap, maxEmIter=1000, maxVarIter=1000, convergence=1e-4, topicNum=None, paraLambda=None):
        """
        init model
        :param wordMap:
        :param topicNum:
        :param paraLambda:
        """
        self.wordMap = wordMap
        self.K = (config.K if topicNum is None else topicNum)
        self.docs = None
        self.mu = None
        self.sigma = None
        self.invSigma = None
        self.paraLambda = (config.paraLambda if paraLambda is None else paraLambda)
        self.oldParaLambda = self.paraLambda
        self.logBeta = None
        self.beta = None
        self.lHood = 0
        self.trainDocs = None
        self.testDocs = None
        self.detInvSigma = 1
        self.maxEMIter = maxEmIter
        self.maxVarIter = maxVarIter
        self.convergence = convergence
        self.initParameters()
        self.saver = FileSaver()

    def initParameters(self):
        self.mu = np.random.uniform(-0.1, 0.1, [self.K])
        mathUtil.simplexNorm(self.mu)
        self.sigma = mathUtil.correlatedMatrixGenerate([self.K, self.K], 1.0)
        self.invSigma = mathUtil.invMatrix(self.sigma)
        self.detInvSigma = np.linalg.det(self.invSigma)
        self.beta = mathUtil.betaGenerate([self.K, self.wordMap.size()])
        self.logBeta = np.log(self.beta)

    def inference(self):
        currentIter = 0
        convergence = 1
        self.docs = self.trainDocs
        while currentIter < self.maxEMIter and convergence > self.convergence:
            self.doEM(self.docs)
            self.doEstimateParameters(self.docs)
            self.pLogLikelihood(self.docs)
            # todo update convergence
            currentIter += 1

    def doEM(self, docs):
        for doc in docs:
            cIter = 0
            convergence = 1
            while cIter < self.maxVarIter and doc.convergence < convergence:
                self.updateGamma(doc)
                self.updateNu(doc)
                for sentence in doc.sentences:
                    self.updateZeta(sentence)
                    self.updateXi(doc, sentence)
                    self.updatePsi(sentence)
                    self.updateOmega(sentence)
                    self.updatePhi(sentence)
                    sentence.updateVar()
                self.qLogLikelihood(doc)
                doc.updateVar()

    def qLogLikelihood(self, doc):
        pass

    def pLogLikelihood(self, docs):
        pass

    def doEstimateParameters(self, docs):
        pass

    def updateGamma(self, doc):
        sumXi = doc.sumXi
        psiSum = np.zeros([self.K], np.float64)
        for sentence in doc.sentences:
            psiSum += sentence.xi*sentence.psi
        _mSigma = np.matrix(self.invSigma, copy=True)
        for k in range(self.K):
            _mSigma[k, k] += sumXi
        _m1 = _mSigma.getI()
        _v1 = self.invSigma*self.mu + psiSum
        doc.gamma = _m1 * _v1

    def updateNu(self, doc):
        sumXi = doc.sumXi
        for i in range(self.K):
            doc.nu[i] = 1/math.sqrt(self.sigma[i, i]+sumXi)

    def updateZeta(self, sentence):
        for k in range(self.K):
            sentence.zeta += math.exp(sentence.psi[k]+0.5*sentence.oldOmega**2)

    def updateXi(self, doc, sentence):
        """
        update \xi in sentence
        :param doc:
        :param sentence:
        :return:
        """
        C = self.K*sentence.oldOmega**2
        C += np.inner(doc.oldNu,doc.oldNu)
        _pg = sentence.psi-doc.oldGamma
        C += np.inner(_pg, _pg)
        C /= 2
        sentence.xi = math.sqrt((2+self.K)**2/(16*C**2)+2*self.paraLambda/C)-(2+self.K)/(4*C)

    def updatePsi(self, sentence):
        # todo updare psi using Newton Method
        pass

    def updateOmega(self, sentence):
        # todo updare psi using Newton Method
        pass

    def updatePhi(self, sentence):
        for n in range(sentence.wordNum):
            for k in range(self.K):
                sentence.oldPhi[k] = math.exp(sentence.psi[k]-1)*self.beta[k,sentence.words[n]]

    def estimate(self):
        self.doEM(self.testDocs.getDocs())

    def setTrainDataset(self, docs):
        self.trainDocs = docs

    def setTestDataset(self, docs):
        self.testDocs = docs

    def saveParameters(self):
        # todo save parameters
        pass

import math

import numpy as np
from CTMSA.src.util import mathUtil
from scipy import optimize as sop
from CTMSA.src import config
from CTMSA.src.util.FileSaver import FileSaver


class CTMSAModel:
    """
    correlated topic model with sentence attention
    """

    def __init__(self, wordMap, maxEmIter=1000, maxVarIter=1000, modelConvergence=1e-4, sConvergence=1e-4,
                 topicNum=None, paraLambda=None):
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
        self.oldMu = None
        self.sigma = None
        self.invSigma = None
        self.oldInvSigma = None
        self.oldSigma = None
        self.paraLambda = (config.paraLambda if paraLambda is None else paraLambda)
        self.oldParaLambda = self.paraLambda
        self.logBeta = None
        self.oldLogBeta = None
        self.beta = None
        self.oldBeta = None
        self.lHood = 0
        self.trainDocs = None
        self.testDocs = None
        self.detInvSigma = 1
        self.oldDetInvSigma = self.detInvSigma
        self.maxEMIter = maxEmIter
        self.maxVarIter = maxVarIter
        self.modelConvergence = modelConvergence
        self.sConvergence = sConvergence
        self.initParameters()
        self.saver = FileSaver()

    def initParameters(self):
        self.mu = np.random.uniform(-0.1, 0.1, [self.K])
        mathUtil.simplexNorm(self.mu)
        self.oldMu = self.mu.copy()
        self.sigma = mathUtil.correlatedMatrixGenerate([self.K, self.K], 1.0)
        self.oldSigma = self.sigma.copy()
        self.invSigma = mathUtil.invMatrix(self.sigma)
        self.oldInvSigma = self.invSigma.copy()
        self.detInvSigma = np.linalg.det(self.invSigma)
        self.oldDetInvSigma = self.detInvSigma
        self.beta = mathUtil.betaGenerate([self.K, self.wordMap.size()])
        self.oldBeta = self.beta.copy()
        self.logBeta = np.log(self.beta)
        self.oldLogBeta = self.logBeta.copy()

    def inference(self):
        currentIter = 0
        convergence = 1.0
        self.docs = self.trainDocs
        while currentIter < self.maxEMIter and convergence > self.modelConvergence:
            self.doEM(self.docs)
            self.doEstimateParameters(self.docs)
            convergence = self.doCorpusLogLikelihood(self.docs)
            # todo update convergence
            currentIter += 1

    def doEM(self, docs):
        for doc in docs:
            cIter = 0
            self.docLogLikelihood(doc)
            while cIter < self.maxVarIter and doc.convergence < self.sConvergence:
                self.updateGamma(doc)
                self.updateNu(doc)
                for sentence in doc.sentences:
                    sIter = 0
                    while sIter < self.maxVarIter and sentence.convergence < self.sConvergence:
                        self.updateZeta(sentence)
                        self.updateXi(doc, sentence)
                        self.updatePsi(sentence)
                        self.updateOmega(sentence)
                        self.updatePhi(sentence)
                        sentence.updateVar()
                        self.sLogLikelihood(sentence)
                        sIter += 1
                self.docLogLikelihood(doc)
                cIter += 1
                doc.updateVar()

    def sLogLikelihood(self, sentence):
        pass

    def docLogLikelihood(self, doc):
        pass

    def doCorpusLogLikelihood(self, docs):
        # todo calculate the corpus convergence
        return 1

    def doEstimateParameters(self, docs):
        pass

    def updateGamma(self, doc):
        sumXi = doc.sumXi
        psiSum = np.zeros([self.K], np.float64)
        for sentence in doc.sentences:
            psiSum += sentence.oldXi * sentence.oldPsi
        _mSigma = np.matrix(self.oldInvSigma, copy=True)
        for k in range(self.K):
            _mSigma[k, k] += sumXi
        _m1 = _mSigma.getI()
        _v1 = self.oldInvSigma * self.oldMu + psiSum
        doc.gamma = _m1 * _v1
        mathUtil.normVec2One(doc.gamma)

    def updateNu(self, doc):
        sumXi = doc.sumXi
        for i in range(self.K):
            doc.nu[i] = 1 / math.sqrt(self.oldSigma[i, i] + sumXi)

    def updateZeta(self, sentence):
        for k in range(self.K):
            sentence.zeta += math.exp(sentence.psi[k] + 0.5 * sentence.oldOmega ** 2)

    def updateXi(self, doc, sentence):
        C = self.K * sentence.oldOmega ** 2
        C += np.inner(doc.oldNu, doc.oldNu)
        _pg = sentence.psi - doc.oldGamma
        C += np.inner(_pg, _pg)
        C /= 2
        sentence.xi = math.sqrt((2 + self.K) ** 2 / (16 * C ** 2) + 2 * self.oldParaLambda / C) - (2 + self.K) / (4 * C)

    def updatePsi(self, doc, sentence):
        # todo updare psi using Conjugate Gradient Algorithm
        global optPsi
        tol = 100.0
        over = False
        # True / False *
        direction = 0
        while not over:
            oldOptPsi = sop.minimize(self.fPsi, x0=sentence.oldPsi, args=(doc, sentence), method='CG', jac=self.dfPsi,
                                  tol=1000)
            if optPsi.success:
                optPsi = oldOptPsi
                if direction > 0:
                    over = True
                else:
                    tol /= 10
                    direction = -1
            else:
                if direction < 0:
                    over = True
                else:
                    direction = 1
                    tol *= 10
        for i in range(doc.K):
            sentence.psi[i] = optPsi.x[i]

    def fPsi(self, psi, doc, sentence):
        tmp = 0.5 * sentence.oldXi * np.vdot(psi - doc.oldGamma, psi - doc.oldGamma)
        for i in range(sentence.wordNum):
            for j in range(doc.K):
                tmp += math.exp(psi[j] + sentence.oldOmega ** 2 / 2) / sentence.oldZeta
                tmp -= sentence.oldPhi[i, j] * psi[j]
        return tmp

    def dfPsi(self, psi, doc, sentence):
        tmp = sentence.oldXi * (psi - doc.oldGamma)
        for i in range(doc.K):
            tmp[i] += sentence.wordNum * math.exp(psi[i] + sentence.oldOmega ** 2) / sentence.oldZeta
            for n in range(sentence.wordNum):
                tmp[i] -= sentence.oldPhi[n, i]
        return tmp

    def updateOmega(self, sentence):
        # todo update psi using Newton Method

        pass

    def updatePhi(self, sentence):
        for n in range(sentence.wordNum):
            for k in range(self.K):
                sentence.phi[k] = math.exp(sentence.oldPsi[k] - 1) * self.oldBeta[k, sentence.words[n]]
        mathUtil.normVec2One(sentence.phi)

    def estimate(self):
        self.doEM(self.testDocs.getDocs())

    def setTrainDataset(self, docs):
        self.trainDocs = docs

    def setTestDataset(self, docs):
        self.testDocs = docs

    def saveParameters(self):
        # todo save parameters
        pass

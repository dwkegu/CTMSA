import numpy as np
from CTMSA.src.model.Sentence import Sentence


class Document:
    def __init__(self, K, rawIdDoc):
        self.K = K
        self.sentenceNum = 0
        self.sentences = []
        self.gamma = np.ndarray([K], np.float64)
        self.oldGamma = self.gamma.copy()
        self.nu = np.ones([K], np.float64)
        self.oldNu = self.nu.copy()
        self.logLikelihood = 0
        self.convergence = 1
        self.initDoc(rawIdDoc)
        self.sumXi = 0
        self.initParameters()

    def initParameters(self):
        left = 1
        for i in range(self.K-1):
            self.gamma[i] = 1/self.K
            self.oldGamma[i] = self.gamma[i]
            left -= self.gamma[i]
            self.nu[i] = 1
            self.oldNu[i] = 1
        self.nu[self.K-1] = 1
        self.oldNu[self.K-1] = 1
        self.gamma[self.K-1] = left
        self.oldGamma[self.K-1] = self.gamma[self.K-1]

    def initDoc(self, rawIdDoc):
        self.sentences.clear()
        self.sumXi = 0
        for line in rawIdDoc:
            s = Sentence(self.K, self, line)
            self.sentences.append(s)
            self.sumXi += s.xi

    def getSentence(self, i):
        if i >= self.sentenceNum or i < 0:
            return None
        return self.sentences[i]

    def updateVar(self):
        for i in range(self.K):
            self.oldGamma[i] = self.gamma[i]
            self.oldNu[i] = self.nu[i]


import numpy as np
from .Sentence import Sentence


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
        self.oldGamma = self.gamma
        self.oldNu = self.nu

import numpy as np


class Sentence:
    def __init__(self, K, doc, content):
        self.wordNum = 0
        self.words = list()
        self.xi = 1.0
        self.oldXi = self.xi
        self.topicAssign = list()
        self.omega = 1.0
        self.oldOmega = 1.0
        # todo init psi
        self.psi = doc.gamma.copy()
        self.oldPsi = self.psi.copy()
        self.phi = np.ndarray([self.wordNum, K], np.float64)
        self.oldPhi = self.phi.copy()
        # todo init zeta
        self.zeta = 1.0
        self.oldZeta = 1.0
        self.initWithIds(content)
        self.K = K

    def initWithIds(self, content):
        pass

    def updateVar(self):
        self.oldOmega = self.omega
        self.oldXi = self.xi
        for i in range(self.psi.shape[0]):
            self.oldPsi[i] = self.psi[i]
        for i in range(self.wordNum):
            for j in range(self.K):
                self.oldPhi[i, j] = self.phi[i, j]
        self.oldZeta = self.zeta


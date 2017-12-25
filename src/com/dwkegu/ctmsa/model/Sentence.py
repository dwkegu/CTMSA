import numpy as np


class Sentence:
    def __init__(self, K, doc, content):
        self.wordNum = 0
        self.words = list()
        self.xi = 1
        self.topicAssign = list()
        self.omega = 1
        self.oldOmega = 1
        # todo init psi
        self.psi = doc.gamma.copy()
        self.oldPsi = self.psi.copy()
        self.phi = np.ndarray([self.wordNum, K], np.float64)
        self.oldPhi = self.phi.copy()
        # todo init zeta
        self.zeta = 1
        self.oldZeta = 1
        self.initWithIds(content)

    def initWithIds(self, content):
        pass

    def updateVar(self):
        pass
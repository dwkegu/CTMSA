import numpy as np


class Sentence:
    def __init__(self, K, doc, content):
        self.wordNum = 0
        self.words = list()
        self.xi = 1
        self.topicAssign = list()
        self.omega = 1
        # todo init psi
        self.psi = doc.gamma.copy()
        self.phi = np.ndarray([self.wordNum, K], np.float64)
        # todo init zeta
        self.zeta = 1
        self.oldZeta = 1
        self.initWithIds(content)

    def initWithIds(self, content):
        pass
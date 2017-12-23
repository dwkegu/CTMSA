import numpy as np


class Sentence:
    def __init__(self, K):
        self.wordNum = 0
        self.words = list()
        self.xi = 1
        self.topicAssign = list()
        self.omega = 1
        self.psi = None
        self.phi = np.ndarray([self.wordNum, K], np.float64)

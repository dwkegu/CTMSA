import numpy as np


class Document:
    def __init__(self, K):
        self.K = K
        self.sentenceNum = 0
        self.sentences = []
        self.gamma = list()
        self.nu = np.ones([K], np.float64)

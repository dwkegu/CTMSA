import os
import datetime
import time


class FileSaver:
    def __init__(self):
        self.filePath = 'f:/tmp/CTMSA/log/'
        self.subDir = ['parameters', 'variation', 'errors']
        self.runTime = str(datetime.date.today()) + time.time()+'/'

    def saveParameters(self, parameterName, content):
        with open(self.filePath+self.runTime+parameterName+".txt", 'r') as f:
            f.write()
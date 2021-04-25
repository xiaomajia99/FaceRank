import os
import sys
import logging

rootPath = os.getcwd() + "/facerank/"
dataPath = "data/"
logPath = "logs/"
modelPath = "model/"


def configLog(logfile = "logger.log"):
    """
    logfile : log file name
    """
    filePath = rootPath + logPath + logfile
    logging.basicConfig(filename=filePath,level=logging.DEBUG,format='%(asctime)s  %(filename)s  %(levelname)s  %(message)s')
    

def dataPreProcess():
    pass

def trainModel():
    pass

def main():
    configLog()
    logging.info("test")
    logging.debug("debug")
    pass

if __name__ == "__main__":
    main()
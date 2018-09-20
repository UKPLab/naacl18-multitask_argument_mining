##############################################################################
# Claudia Schulz
# November 2017
#
# split datasets into train (70%), development (30%), and placeholder test sets
# (to be used as auxiliary tasks in the MTL setup)
##############################################################################

import os
import random
import logging
from util.preprocessing import perpareDataset

##############################################################################
def createSplits(dataFolder, resultFolder):
    with open(resultFolder + "/train.txt", 'w') as train,\
        open(resultFolder + "/dev.txt", 'w') as dev,\
        open(resultFolder + "/test.txt", 'w') as test:
        # create list of all available datapoints (annotated files)
        dataList = []
        tokens = 0
        for data in os.listdir(dataFolder):
            dataList.extend([data])
            with open(dataFolder + "/" + data, 'r') as dataPoint:
                lines = dataPoint.readlines()
                tokens = tokens + len(lines)

        # shuffle the datapoints
        print(tokens)
        random.shuffle(dataList)
        # determine training and dev size
        trainSize = tokens*0.70

        dataPointCounter = 0
        tokenCounterTrain = 0
        while tokenCounterTrain < trainSize:
            dataPoint = dataList[dataPointCounter]
            with open(dataFolder + "/" + dataPoint, 'r') as text:
                for line in text:
                    train.write(line)
                    tokenCounterTrain +=1
            train.write("\n")
            dataPointCounter +=1

        dataFilesForTraining = dataPointCounter + 1
        print("Train set of " + resultFolder + " has size (no of tokens): " + str(tokenCounterTrain) + "\n")
        print("This used " + str(dataFilesForTraining) + " data files out of " + str(len(dataList)) + "\n")

        # write 1 datapoint as test
        tokenCounterTest = 0
        dataPointTrain = dataList[dataPointCounter]
        with open(dataFolder + "/" + dataPointTrain, 'r') as text:
            for line in text:
                test.write(line)
                tokenCounterTest +=1
        test.write("\n")
        dataPointCounter += 1

        dataFilesForTest = dataPointCounter + 1 - dataFilesForTraining
        print("Test set of " + resultFolder + " has size (no of tokens):" + str(tokenCounterTest) + "\n")
        print("This used 1 data file out of " + str(len(dataList)) + "\n")

        # write the rest as dev set
        tokenCounterDev = 0
        while dataPointCounter < len(dataList):
            dataPoint = dataList[dataPointCounter]
            with open(dataFolder + "/" + dataPoint, 'r') as text:
                for line in text:
                    dev.write(line)
                    tokenCounterDev +=1
            dev.write("\n")
            dataPointCounter +=1

        dataFilesForDev = dataPointCounter + 1 - dataFilesForTraining - dataFilesForTest
        print("Dev set of " + resultFolder + " has size (no of tokens): " + str(tokenCounterDev) + "\n")
        print("This used " + str(dataFilesForDev) + " data files out of " + str(len(dataList)) + "\n")

        print("In total, " + str(dataPointCounter) + " data files out of "+ str(len(dataList)) + " were used")

        return


##############################################################################
##############################################################################

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

#ch = logging.StreamHandler(sys.stdout)
ch = logging.FileHandler('fullDataArgExperiments.log')
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# USER ACTION NEEDED
# put BIO-labelled corpora here (in subfolders)
dataPath = dname + "/corpora"

# output path for train, dev, test splits (subfolders will be created for each corpus)
resultPath = dname + "/data_fullSplits"

for dataName in os.listdir(dataPath):
    dataFolder = dataPath + "/" + dataName
    if os.path.isdir(dataFolder):
        resultFolder = resultPath + "/" + dataName
        if not os.path.exists(resultFolder):
            os.makedirs(resultFolder, 0o777)
        createSplits(dataFolder, resultFolder)



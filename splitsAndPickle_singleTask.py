##############################################################################
# Claudia Schulz
# November 2017
#
# take 30,000 tokens of each dataset and split this into 21,000 train and 9,000 dev (70%,30%)
# take the rest of the dataset for test
# and create pickled files for each dataset with different word embeddings
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
        for data in os.listdir(dataFolder):
            dataList.extend([data])
        #print(dataList)
        # shuffle the datapoints
        random.shuffle(dataList)
        dataPointCounter = 0

        # write (at least) 21000 tokens as training set
        tokenCounterTrain = 0
        while tokenCounterTrain <= 21000:
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


        # write (at least) 9000 tokens as dev set
        tokenCounterDev = 0
        while tokenCounterDev <= 9000:
            dataPoint = dataList[dataPointCounter]
            with open(dataFolder + "/" + dataPoint, 'r') as text:
                for line in text:
                    dev.write(line)
                    tokenCounterDev +=1
            dev.write("\n")
            dataPointCounter +=1

        dataFilesForDev = dataPointCounter + 1 - dataFilesForTraining
        print("Dev set of " + resultFolder + " has size (no of tokens): " + str(tokenCounterDev) + "\n")
        print("This used " + str(dataFilesForDev) + " data files out of " + str(len(dataList)) + "\n")

        # write the rest as test set
        tokenCounterTest = 0
        while dataPointCounter < len(dataList):
            dataPoint = dataList[dataPointCounter]
            with open(dataFolder + "/" + dataPoint, 'r') as text:
                for line in text:
                    test.write(line)
                    tokenCounterTest +=1
            test.write("\n")
            dataPointCounter +=1

        dataFilesForTesting = dataPointCounter + 1 - dataFilesForTraining - dataFilesForDev
        print("Test set of " + resultFolder + " has size (no of tokens): " + str(tokenCounterTest) + "\n")
        print("This used " + str(dataFilesForTesting) + " data files out of " + str(len(dataList)) + "\n")

        print("In total, " + str(dataPointCounter) + " data files out of "+ str(len(dataList)) + " were used")
        return


##############################################################################
def pickleData(embeddingsPath, datasetName, dataColumns):
    datasetFiles = [(datasetName, dataColumns), ]
    pickleFile = perpareDataset(embeddingsPath, datasetFiles)



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
ch = logging.FileHandler('singleArgExperimentsSmallData.log')
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# USER ACTION NEEDED
# put BIO-labelled corpora here (in subfolders)
dataPath = dname + "/corpora"

# output path for train, dev, test splits (subfolders will be created for each corpus)
resultPath = dname + "/data_singleTask"

# USER ACTION NEEDED
# put embeddings here (GloVe and [Komninos & Mandhar 2016])
embeddingsPath = dname + "/embeddings"


for dataName in os.listdir(dataPath):
    dataFolder = dataPath + "/" + dataName
    print(dataName)
    if os.path.isdir(dataFolder):
        resultFolder = resultPath + "/" + dataName
        if not os.path.exists(resultFolder):
            os.makedirs(resultFolder, 0o777)
        createSplits(dataFolder, resultFolder)
        for embeddingsName in os.listdir(embeddingsPath):
            if embeddingsName == "glove.txt" or embeddingsName == "wiki_extvec":
                embeddingsFull = embeddingsPath + "/" + embeddingsName
                if os.path.isfile(embeddingsFull):
                    print(embeddingsName)
                    pickleData(embeddingsFull, dataName, {0:'tokens', 1:'arg_BIO'})


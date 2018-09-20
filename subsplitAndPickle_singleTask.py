##############################################################################
# Claudia Schulz
# December 2017

# reduce given small train sets further
# --> only train set is reduced, development and test stays the same
# and create pickled files for each dataset with 2 different word embeddings
#
# NOTE: must have dev and test set in data folder already!
##############################################################################

import os
import random
import logging
from util.preprocessing import perpareDataset

##############################################################################
def createSubSplits(dataFolder, resultFolder, trainSize):
    with open(resultFolder + "/train.txt", 'w') as train,\
        open(dataFolder + "/train.txt", 'r') as givenTrain:

        dataPointCounter = 0

        # write (at least) trainSize tokens as training set

        tokenCounterTrain = 0
        for line in givenTrain:
            if line == "\n":
                dataPointCounter +=1
            if tokenCounterTrain <= trainSize or line != "\n":
                train.write(line)
                tokenCounterTrain += 1
            else:
                break

        print("Train set of " + resultFolder + " has size (no of tokens): " + str(tokenCounterTrain) + "\n")
        print("This used " + str(dataPointCounter) + " documents\n")

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
ch = logging.FileHandler('singleArgExperiments_subsplits.log')
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# USER ACTION NEEDED
# put the splits (only train.txt required) created with splitsAndPickle.py here - in subfolders for each corpus
# if needed, change path name
dataPath = dname + "/givenData"

# USER ACTION NEEDED
# put test.txt and dev.txt created with splitsAndPickle_singleTask.py here - in subfolders for each corpus
# the smaller train split will then be added
resultPath = dname + "/data_singleTask_1000"

# USER ACTION NEEDED
# put embeddings here (GloVe and [Komninos & Mandhar 2016])
embeddingsPath = dname + "/embeddings"

# USER ACTION NEEDED
# specify size of training set here
newTrainSize = 1000


for dataName in os.listdir(dataPath):
    dataFolder = dataPath + "/" + dataName
    print(dataName)
    if os.path.isdir(dataFolder):
        resultFolder = resultPath + "/" + dataName
        if not os.path.exists(resultFolder):
            os.makedirs(resultFolder, 0o777)
        createSubSplits(dataFolder, resultFolder, newTrainSize)
        for embeddingsName in os.listdir(embeddingsPath):
            if embeddingsName == "glove.txt" or embeddingsName == "wiki_extvec":
                embeddingsFull = embeddingsPath + "/" + embeddingsName
                if os.path.isfile(embeddingsFull):
                    print(embeddingsName)
                    pickleData(embeddingsFull, dataName, {0:'tokens', 1:'arg_BIO'})

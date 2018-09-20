##############################################################################
# Claudia Schulz
# December 2017
#
# create pickled files for each dataset with different word embeddings
##############################################################################

import os
import logging
import random
from util.preprocessing import perpareDataset



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

ch = logging.FileHandler('singleArgExperiments.log')
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# USER ACTION NEEDED
# put train, test, dev here - in subfolders for each corpus
# if needed, change path name
dataPath = dname + "/dataSplits/1k"

# USER ACTION NEEDED
# put embeddings here (GloVe and [Komninos & Mandhar 2016])
embeddingsPath = dname + "/embeddings"

for dataName in os.listdir(dataPath):
    dataFolder = dataPath + "/" + dataName
    print(dataName)
    for embeddingsName in os.listdir(embeddingsPath):
        if embeddingsName == "glove.txt" or embeddingsName == "wiki_extvec":
            embeddingsFull = embeddingsPath + "/" + embeddingsName
            if os.path.isfile(embeddingsFull):
                print(embeddingsName)
                pickleData(embeddingsFull, dataName, {0:'tokens', 1:'arg_BIO'})

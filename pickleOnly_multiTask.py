##############################################################################
# Claudia Schulz
# November 2017
#
# use dataset splits of a specified smallData with the dataset splits of the
# the splits created by splitsFullData.py for the other datasets
# and create pickled files  with different word embeddings
# ##############################################################################

import os
import logging
from util.preprocessing import perpareDataset

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
ch = logging.FileHandler('multiArgExperiments.log')
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# USER ACTION NEEDED
# specify name of small data set (target data!)
dataSmallName = "Stab201X"

dataSmallColumns = dataSmallName + 'TARGET_BIO'
dataSetFiles = [(dataSmallName,{0:'tokens', 1:dataSmallColumns})]

# USER ACTION NEEDED
# put embeddings here (GloVe and [Komninos & Mandhar 2016])
embeddingsPath = dname + "/embeddings"

# USER ACTION NEEDED
# make sure that dataPath contains the small train,dev,test files for dataName and
# the full size train,dev,test for all other datasets (as created by splitsFullData.py)
# if needed, change path name
dataPath = dname + "/data_multiTask"

for dataAllName in os.listdir(dataPath):
    if dataAllName != dataSmallName:
        dataAllColumns = dataAllName + '_BIO'
        dataSetFiles.append((dataAllName,{0:'tokens', 1:dataAllColumns}))
        print(dataSetFiles)

for embeddingsName in os.listdir(embeddingsPath):
    if embeddingsName == "glove.txt" or embeddingsName == "wiki_extvec":
        embeddingsFull = embeddingsPath + "/" + embeddingsName
        if os.path.isfile(embeddingsFull):
            print(embeddingsName)
            pickleFile = perpareDataset(embeddingsFull, dataSetFiles)


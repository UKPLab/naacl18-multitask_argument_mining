##############################################################################
# Claudia Schulz
# October 2017
# 
# train multi task LSTM using command line parameters
##############################################################################

from __future__ import print_function
import os
import logging
import sys
from neuralnets.MultiTaskLSTM import MultiTaskLSTM
from util.preprocessing import loadDatasetPickle


# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# get command line arguments for this run (note that 0 is the name of the script and that all arguments are strings)
pickledData = sys.argv[1] # e.g. pathToFile/pickled_dataName1_dataName2..._glove.pkl
datasetName = sys.argv[2] # e.g. dataName (the name of the target data)
dropoutStr = sys.argv[3] # value between 0 and 1, or for variational dropout with 2 values between 0 and 1, e.g. 0.25,0.25
dropout = [float(s) for s in dropoutStr.split(',')]
layersStr = sys.argv[4] # enumeration of neurons per layer of the LSTM, e.g. 100,75,50
layers = [int(s) for s in layersStr.split(',')]
optimizer = sys.argv[5] # one of: sgd, adagrad, adadelta, rmsprop, adam, nadam (we always chose nadam for our experiments)
charEmbedding = sys.argv[6] # one of: None, cnn, lstm (we always chose None for our experiments)
resultsPath = sys.argv[7] # e.g. pathToFile/experimentName_results.txt (prints the F1 scores on dev and test set for each epoch)
detailedPath = sys.argv[8] # e.g. pathToFile/experimentName_detailedResults.txt (prints the prections in each epoch)

#Parameters of the network
params = {'dropout': dropout, 'classifier': ['CRF'], 'LSTM-Size': layers, 'optimizer': optimizer, 'charEmbeddings': charEmbedding, 'miniBatchSize': 32, 'detailedOutput': detailedPath, 'mainModelName': datasetName}

######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset from the already created pickle file
embeddings, word2Idx, datasets = loadDatasetPickle(pickledData)

datasetTuples = {}

# USER ACTION NEEDED
# replace dataset1,... with the actual names of all your datasets used in the multi-task experiment
for data in ['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6']:
    if data == datasetName:
        datasetTuples[data] = (datasets[data], data + 'TARGET_BIO', True)
    else:
        datasetTuples[data] = (datasets[data], data + '_BIO', True)


model = MultiTaskLSTM(embeddings, datasetTuples, params=params)

model.verboseBuild = True
#model.modelSavePath = "models/%s/[DevScore]_[TestScore]_[Epoch].h5" % modelName #Enable this line to save the model to the disk
model.storeResults(resultsPath)

# number is the batch size
model.evaluate(50)

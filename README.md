# Multi-Task Learning for Argumentation Mining in Low-Resource Settings

The following repository contains the code for training single and multi-task LSTMs for argument component identification. The code allows to replicate the experiments of [our paper](http://aclweb.org/anthology/N18-2006).

# Citation 
If you find the implementation useful, please cite the following paper:

```
@inproceedings{Schulz:2018:NAACL,
	title = {Multi-Task Learning for Argumentation Mining in Low-Resource Settings},
	author = {Schulz, Claudia and Eger, Steffen and Daxenberger, Johannes and Kahse, Tobias and Gurevych, Iryna},
	publisher = {Association for Computational Linguistics},
	booktitle = {Proceedings of the 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
	pages = {35--41},
	month = jun,
	year = {2018},
	location = {New Orleans, USA}
}
```
> **Abstract:** We investigate whether and where multitask learning (MTL) can improve performance on NLP problems related to argumentation mining (AM), in particular argument component identification. Our results show that MTL performs particularly
well (and better than single-task learning) when little training data is available for the main task, a common scenario in AM. Our findings challenge previous assumptions that conceptualizations across AM datasets are divergent and that MTL is difficult for semantic or higher-level tasks.


Contact person: Claudia Schulz, clauschulz1812 AT_usual_GMAIL_ending

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 


## Project structure
For the single and multi-task architectures, we used the implementation of Nils Reimers (NR): https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf
The code has been updated since - here you find the original code as used in our experiments:
* neuralnets -- contains BiLISTM.py for single task experiments and MultiTaskLSTM.py for multi-task experiments (slight modification of code by NR to use scikit-learn F1 score)
* util -- various scripts for processing data and other utilities by NR
* dataSplits -- the train, dev, test splits of all datasets used for our experiments


## Experimental setup
All code is run using Python 3.

### Setup with virtual environment (Python 3)

Setup a Python virtual environment (optional):
``` 
virtualenv --system-site-packages -p python3 env
source env/bin/activate
```

Install the requirements:
```
.env/bin/pip3 install -r requirements.txt
```


### Use our data splits and prepare the data
> If you use the data in your experiments, please cite both our NAACL 2018 paper (see above) and the respective papers that first introduced the data (these can be found in our paper).

* single-task experiments: Specify the respective 1k, 6k, 12k, or 21k folder in 'dataSplits' in pickleOnly_singleTask.py and run it. This creates pickle files for all corpora (given our splits) for the specified small data scenario.
* multi-task exeperiments: Create a folder 'data_multiTask' with a subfolder for each dataset. The folder of the main AM task contains the splits from one of the small data scenarios (e.g. from dataSplits/1k/essays) and the 5 folders of the auxiliary AM tasks contain the splits from the respecite 'fullData' folders. Run pickleOnly_multiTask.py to create a pickle file for the main AM task in the respective small data scenario.


### Create your own data splits and prepare the data
(obtain datasets, all of them are freely available and make sure they are all in the correct (2-column) token-BIO_class format)
1) add datasets (BIO-labelled) to the 'corpora' folder - create a subfolder for each dataset
2) add embedding files to the 'embeddings' folder (if later there is a problem with processing the embedding files, the first line of the embedding files may have to be removed - use embeddingFirstLine.py)
3) create datasplits and pickle file with train size 21k tokens using splitsAndPickle_singleTask.py (automatically done for each dataset in 'corpora')
4) create smaller test sets from the existing ones for 12k, 6k, and 1k tokens using subsplitAndPickle_singleTask.py (automatically done for each dataset)  and create pickle file
5) create train/dev split with full size of each dataset in 'corpora' to be used for training the auxiliary AM tasks in the MTL setup using splitsFullData.py
6) create pickle file for MTL setup using pickleOnly_multiTask.py: the dataPath needs to contain a subfolder for each dataset, where the folder of the main AM task contains the splits from 3) or 4) and the folders of the auxiliary AM tasks contain the splits from 5)

In all scripts, we specify where the user has to adapt the code (mostly file paths) with 'USER ACTION NEEDED'

### Run the Experiments
* singleTaskMain.py runs a BiLSTM with given command line parameters
* multiTaskMain.py runs a multi-task BiLSTM with given command line parameters

Example for running the code:
```
python3 singleTaskMain.py pkl/nameOfMyPickledFile myArgDataset arg_BIO 0.25,0.25 100,50 nadam None results/experimentName.txt detailedOutput/experimentName.txt
```
'arg_BIO' should not be changed when using the data processing as described before, all other parameters can be adapted, the file paths need to be changed

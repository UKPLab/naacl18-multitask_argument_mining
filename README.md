# Multi-Task Learning for Argumentation Mining in Low-Resource Settings

The following repository contains the code for training single and multi-task LSTMs for argument component identification. 

# Citation 
If you find the implementation useful, please cite the following paper:

```
@inproceedings{Schulz:2018:NAACL,
	title = {Multi-Task Learning for Argumentation Mining in Low-Resource Settings},
	author = {Schulz, Claudia and Eger, Steffen and Daxenberger, Johannes and Kahse, Tobias and Gurevych, Iryna},
	publisher = {Association for Computational Linguistics},
	booktitle = {Proceedings of the 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
	pages = {to appear},
	month = jun,
	year = {2018},
	location = {New Orleans, USA}
}
```
> **Abstract:** We investigate whether and where multitask learning (MTL) can improve performance on NLP problems related to argumentation mining (AM), in particular argument component identification. Our results show that MTL performs particularly
well (and better than single-task learning) when little training data is available for the main task, a common scenario in AM. Our findings challenge previous assumptions that conceptualizations across AM datasets are divergent and that MTL is difficult for semantic or higher-level tasks.


Contact person: Claudia Schulz, schulz@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 


## Project structure
For the single and multi-task architectures, we used the implementation of Nils Reimers (NR): https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf
The code has been updated since - here you find the original code as used in our experiments:
* neuralnets -- contains BiLISTM.py for single task experiments and MultiTaskLSTM.py for multi-task experiments (slight modification of code by NR to use scikit-learn F1 score)
* util -- various scripts for processing data and other utilities by NR




# 🎪 Life is a Circus and We are the Clowns: Automatically Finding Analogies between Situations and Processes 🤡
This repository contains the code for the paper.

## Setup
The code is implemented in python 3.8.12. To run it, please install the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Where to start?
Explore the **paper_experiments_results** folder for restoring the results in the experiment 
(each folder contains a separate README file). <br/>
Run **runner.py** for running our algorithm on a specific example of pairs of texts. <br/>
Note that you don't need to run coreference and qa_srl, as the output files have already exist in the repo. 
(You should run coreference and qa_srl only if you use a new input text files, 
by setting run_coref=False, run_qasrl=False in analogous_matching_algorithm function)

## Important folders

**paper_experiments_results**:<br/>
Contains the datasets, the labels of the annotators, as well as the data which generates the results in the figures 
and tables of the three experiments. Each inner folder contains a separate README file.<br/>

**data:**<br/>
Includes the following folders:<br/>
**original_text_files** -- all the original texts files (including the stories and paragraphs from ProPara).<br/>
**coref_text_files** -- all the texts files after coreference (including the stories and paragraphs from ProPara).<br/>
**propara** -- data files relevant to ProPara dataset, output files of the ranking lists for the different models 
   (see Section 4.1 in the paper), and some code files to read and print stats on ProPara and the methods.<br/>
   
**s2e-coref:**<br/>
Contains the implementation code for the coreference model that we used (see Section 3.1 in the paper).<br/>

**qasrl-modeling**<br/>
Contains the implementation code for the QA-SRL model that we used (see Section 3.2 in the paper).<br/>

## Important py. files

## Algorithm's code files
**runner.py** -- runner of our analogous matching algorithm on given pairs.<br/>
**find_mappings.py** -- run FMQ method on a given pair of texts  (called from outside to generate_mappings function).<br/>
**find_mappings_verbs.py** -- run FMV method on a given pair of texts (called from outside to generate_mappings function).\
**sentence_bert.py** -- run SBERT on a given pair of texts.<br/>
**coref.py** -- run our coreference implementation on input files.<br/>
**qa-srl.py** -- run our QA-SRL implementation on texts files (after coref).<br/>

## Experiment's code files
**run_propara_all_pairs_exp.py** -- run experiment 1 (see Section 4.1 in the paper).<br/>
**analogies_mining_exp_annotators_consistency.py** -- run annotators consistency confusion matrix 
(see Section 4.1 in the paper).<br/>
**run_mappings_evaluation_exp.py** -- run experiment 2 (see Section 4.2 in the paper).<br/>
**run_robustness_to_paraphrases_exp.py** -- run experiment 3 (see Section 4.3 in the paper).<br/>

## Cite
TODO

## Contact
For inquiries, please send an email to oren.sultan@mail.huji.ac.il.
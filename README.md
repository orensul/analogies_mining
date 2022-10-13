# Life is a Circus and We are the Clowns: Automatically Finding Analogies between Situations and Processes

# This repository contains the code for the paper.

## important folders:

## paper_experiments_results:
Contains the datasets as well as the data which generates the results in the figures and tables of the three experiments. 
Each inner folder contains a separate README file.

## data:
# Includes the following folders:
# original_text_files -- all the original texts files (including the stories and paragraphs from ProPara)
# coref_text_files -- all the texts files after coreference (including the stories and paragraphs from ProPara)
# propara -- data files relevant to ProPara dataset, output files of the ranking lists for the different models 
   (see Section 4.1 in the paper), and some code files to read and print stats on ProPara and the methods.
   
## s2e-coref
Contains the implementation code for the coreference model that we used (see Section 3.1 in the paper).

## qasrl-modeling
Contains the implementation code for the QA-SRL model that we used (see Section 3.2 in the paper).

## important py. files:
## algorithm code files:
# runner.py -- runner of our analogous matching algorithm on given pairs.
# find_mappings.py -- run FMQ method on a given pair of texts  (called from outside to generate_mappings function)
# find_mappings_verbs.py -- run FMV method on a given pair of texts (called from outside to generate_mappings function)
# sentence_bert.py -- run SBERT on a given pair of texts.
# coref.py -- run coreference on input files (NO NEED TO TOUCH IT UNLESS YOU USE NEW DATA)
# qa-srl.py -- run QA-SRL on texts files (after coref) (NO NEED TO TOUCH IT UNLESS YOU USE NEW DATA)

## experiment code files:
# run_propara_all_pairs_exp.py -- run experiment 1 (see Section 4.1 in the paper)
# analogies_mining_exp_annotators_consistency.py -- run annotators consistency confusion matrix (see Section 4.1 in the paper)
# run_mappings_evaluation_exp.py -- run experiment 2 (see Section 4.2 in the paper)
# run_robustness_to_paraphrases_exp.py -- run experiment 3 (see Section 4.3 in the paper)





# Life is a Circus and We are the Clowns: Automatically Finding Analogies between Situations and Processes

This repository contains the code for the paper.

# important folders:

#* paper_experiments_results:
Contains the datasets as well as the data for the results in the figures and tables of the three experiments. Each inner folder contains a separate README file.

#* data:
Includes the following folders:
1) original_text_files -- all the original texts files (including the stories and paragraphs from ProPara)
2) coref_text_files -- all the texts files after coreference (including the stories and paragraphs from ProPara)
3) propara -- data files relevant to ProPara dataset, output files of the ranking lists for the different models 
   (exp1) and some code files to read and print stats on ProPara and the methods.
   
#* s2e-coref
Contains the implementation code for the coreference model that we used (see Section 3.1 in the paper)

#* qasrl-modeling
Contains the implementation code for the QA-SRL model that we used (see Section 3.2 in the paper)


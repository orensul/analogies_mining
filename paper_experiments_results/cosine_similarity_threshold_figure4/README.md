**find_mappings_questions_cosine_sim.txt** and **find_mappings_verbs_cosine_sim.txt**
contains questions / verbs from running ProPara pairs of paragraphs with the similarity score. <br/>
  
**samples_for_questions_cosine_threshold.csv** and **samples_for_verbs_cosine_threshold.csv**
contains samples from the previous files in different range of thresholds, with label 1 if the two questions / verbs 
should be similar and label 0 otherwise. <br/>
  
**read_questions_cosine_sim.py** and **read_verbs_cosine_sim.py** create the plots in the 
appendix (Figure4) by using the tagging results on the previous files. <br/>
  
As a reminder -- we got to the conclusion that the best cosine similarity threshold is 
0.7 for the questions (FMQ) and 0.5 for the verbs (FMV).  
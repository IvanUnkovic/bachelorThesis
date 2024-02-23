# bachelorThesis
Repository containing scripts I used to write my bachelor thesis.

The most important file is finetuning_model.py, which holds all the information regarding removing layers, training the model and doing the finetuning. finetuning_ablation.py is an older version of that. 
Next, call_sigmoid.py is responsible for calling the sigmoid function on anything that the classification head sends. It takes a number and transforms it to a number between 0 and 1. 
Lastly, calculate_metrics.py is a file used for model evaluation. It calculates accuracy, precision, recall, specificity, f1 score, AUC(precision-recall), AUC(ROC) and addittionaly, the confusion matix.

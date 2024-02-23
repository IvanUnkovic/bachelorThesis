# bachelorThesis
Repository containing scripts I used to write my bachelor thesis.

These are the scripts that were used for training and evaluating Rockfish models.

The most important file is finetuning_model.py, which holds all the information regarding removing layers, training the model and doing the finetuning. "remove_layer" function is used for removing specific layer/s from a decoder/encoder, "load_for_finetune" loads the model and prepares it for training by freezing some layers, and "finetune_main" does the training of the model. Name of the checkpoint produced through training, can and should be modified. 

finetuning_ablation.py is just an older version of finetuning_model.py. 

Next, call_sigmoid.py is responsible for calling the sigmoid function on anything that the classification head sends. It takes a number and transforms it to a number between 0 and 1. 

Lastly, calculate_metrics.py is a file used for model evaluation. It calculates accuracy, precision, recall, specificity, f1 score, AUC(precision-recall), AUC(ROC) and addittionaly, the confusion matix.

graph_script.py is only used for plotting graphs.

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve

def calculate_metrics(test_file, predictions_file):

    with open(test_file, 'r') as test_file:
        test_data = test_file.readlines()

    with open(predictions_file, 'r') as predictions_file:
        predictions_data = predictions_file.readlines()

    test_class_labels = {}
    true_labels = []
    predicted_labels = []
    probabilities = []

    for line in test_data:
        line = line.strip().split('\t')
        chromosome = line[1]
        position = line[2]
        class_label = line[3]
        test_class_labels[(chromosome, position)] = class_label

    for line in predictions_data:
        line = line.strip().split('\t')
        chromosome = line[0]
        start = line[1]
        end = line[2]
        probability = float(line[3])

        prediction = '1' if probability >= 0.5 else '0'

        test_class_label = test_class_labels.get((chromosome, start), None)

        if test_class_label is not None:
            true_labels.append(test_class_label)
            predicted_labels.append(prediction)
            probabilities.append(probability)

    true_labels = np.array(true_labels, dtype=int)
    predicted_labels = np.array(predicted_labels, dtype=int)

    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(true_labels, probabilities)
    pr_auc = auc(recall, precision)

    # plot ROC curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (AUC = {:.2f})'.format(pr_auc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

    accuracy = np.sum(np.array(true_labels) == np.array(predicted_labels)) / len(true_labels) * 100
    cm = confusion_matrix(true_labels, predicted_labels)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) * 100
    sensitivity = tp / (tp + fn) * 100
    precision = tp / (tp + fp) * 100
    f1_score = 2 * ((precision * sensitivity) / (precision + sensitivity))

    return accuracy, specificity, sensitivity, precision, f1_score, cm, roc_auc, pr_auc

test_labels_file = sys.argv[1]
predictions_file = sys.argv[2]
accuracy, specificity, sensitivity, precision, f1_score, cm, roc_auc, pr_auc = calculate_metrics(test_labels_file, predictions_file)

print("Accuracy: {:.2f}%".format(accuracy))
print("Specificity: {:.2f}%".format(specificity))
print("Sensitivity: {:.2f}%".format(sensitivity))
print("Precision: {:.2f}%".format(precision))
print("F1 score: {:.2f}".format(f1_score))
print("Confusion Matrix:")
print(cm)
print("AUC of ROC Curve: {:.2f}".format(roc_auc))
print("AUC of Precision-Recall Curve: {:.2f}".format(pr_auc))

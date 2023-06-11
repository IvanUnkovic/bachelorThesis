import sys
from sklearn.metrics import confusion_matrix

def calculate_metrics(test_file, predictions_file):

    with open(test_file, 'r') as test_file:
        test_data = test_file.readlines()

    with open(predictions_file, 'r') as predictions_file:
        predictions_data = predictions_file.readlines()

    test_class_labels = {}
    true_labels = []
    predicted_labels = []
    total_counter = 0
    correct_counter = 0

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
            if test_class_label == prediction:
                correct_counter += 1
            total_counter +=1

    accuracy = correct_counter / total_counter * 100
    cm = confusion_matrix(true_labels, predicted_labels)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) * 100
    sensitivity = tp / (tp + fn) * 100
    precision = tp / (tp + fp) * 100 
    return accuracy, specificity, sensitivity, precision, cm

    

test_labels_file = sys.argv[1]
predictions_file = sys.argv[2]
accuracy, specificity, sensitivity, precision, cm = calculate_metrics(test_labels_file, predictions_file)
print("Accuracy: {}".format(accuracy))
print("Specificity: {}".format(specificity))
print("Sensitivity: {}".format(sensitivity))
print("Precision: {}".format(precision))
print("Confusion Matrix:")
print(cm)


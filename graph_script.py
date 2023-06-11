import numpy as np
import matplotlib.pyplot as plt

#name of the models are defined based on a specific comparison
models = []
metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity']

#values of metrics are obtained through calculate_metrics.py
accuracy = []
precision = []
recall = []
specificity = []

bar_width = 0.2
index = np.arange(len(models))

fig, ax = plt.subplots()
rects = []
for i, metric in enumerate(metrics):
    x = index + (i * bar_width)
    rects.append(ax.bar(x, eval(metric.lower()), bar_width))

ax.set_xlabel('Models')
ax.set_ylabel('Percentage')
ax.set_xticks(index + (bar_width * (len(metrics) - 1)) / 2)
ax.set_xticklabels(models)
ax.legend(rects, metrics)


#a specific layout of y-axis is created by ylim()
plt.ylim()
plt.tight_layout()
plt.show()
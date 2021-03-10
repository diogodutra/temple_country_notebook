import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import product


def plot_confusion_matrix(y_true, y_pred, labels=None):
  """Plots the confusion matrix.
  
  Args:
    y_true (NumPy array): Ground truth (correct) target values
    y_pred (NumPy array): Estimated targets as returned by a classifier
    labels (list of str, optional): List of labels to index the matrix
  """
  cm = confusion_matrix(y_true, y_pred)
  
  if labels is None:
    ticks = list(range(cm.shape[0]))
  else:
    ticks = range(len(labels))
    
  plt.yticks(ticks=ticks, labels=labels)
  plt.xticks(ticks=ticks, labels=labels, rotation=90)
  
  plt.gcf().patch.set_facecolor('white')
  plt.ylabel('True')
  plt.xlabel('Predicted')
  plt.title('Confusion Matrix')
  plt.imshow(cm, cmap='Blues')
  
  # annotate
  for i, j in product(ticks, ticks):
      plt.gca().text(j, i, cm[i, j],
            ha="center", va="center", color="k")
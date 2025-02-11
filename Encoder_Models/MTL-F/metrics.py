# Import Libraries
import torch
from sklearn.metrics import classification_report, confusion_matrix

# Define the classification report function
def get_classification_report(predicted, true, dict=False):
    report = classification_report(true, predicted, digits=4, output_dict=dict)
    return report

# Define the confusion_matrix function
def get_confusion_matrix(predicted, true):
    matrix = confusion_matrix(true, predicted)
    return matrix
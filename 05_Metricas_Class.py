import numpy as np
from sklearn.datasets import load_iris

def load_dataset():
    iris = load_iris()
    x = iris.data
    y = iris.target
    return x, y

def predict_randomly(x_size, num_classes = 3, seed = 42):
    np.random.seed(seed=seed)
    return np.random.randint(num_classes, size=x_size)

def compute_confusion_matrix(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    confusion_matrix = np.zeros((num_classes, num_classes))
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[true_label][pred_label] += 1
    return confusion_matrix
    
def compute_precision(confusion_matrix):
        tp = np.diag(confusion_matrix)
        fp = np.sum(confusion_matrix, axis = 0) - tp
        precision = np.mean(tp/(tp + fp))
        return precision
    
def compute_recall(confusion_matrix):
    tp = np.diag(confusion_matrix)
    fn = np.sum(confusion_matrix, axis=1) - tp
    recall = np.mean(tp/(tp+fn))
    return recall

def compute_f1_score(presicion, recall):
    return 2*(precision*recall) / (precision+recall)

x, y = load_dataset()

y_pred = predict_randomly(x_size=x.shape[0])

confusion_matrix = compute_confusion_matrix(y,y_pred)
print(confusion_matrix)

precision = compute_precision(confusion_matrix)
print(precision)

recall = compute_recall(confusion_matrix)
print(recall)

f1 = compute_f1_score(precision, recall)
print(f1)
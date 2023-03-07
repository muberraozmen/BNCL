from skmultilearn.adapt import MLkNN
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from scipy import sparse
from pandas import DataFrame
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.metrics import label_ranking_average_precision_score, zero_one_loss, coverage_error, label_ranking_loss
import numpy as np


def one_error(targets, scores):
    error = 0
    for i in range(targets.shape[0]):
        row = scores[i]
        if targets[i,np.argmax(scores[i])] != 1:
            error += 1
    error /= targets.shape[0]
    return error
def evaluation(targets, predictions):
    acc = accuracy_score(targets, predictions)
    ha = 1 - hamming_loss(targets, predictions)
    ebf1 = f1_score(targets, predictions, average='samples', zero_division=0)
    mif1 = f1_score(targets, predictions, average='micro', zero_division=0)
    maf1 = f1_score(targets, predictions, average='macro', zero_division=0)
    performance = {'ACC': acc, 'HA': ha, 'ebF1': ebf1, 'miF1': mif1, 'maF1': maf1}
    return performance
def repro_evaluation(targets, results, scores):

    ha = hamming_loss(targets, results)
    one = one_error(targets, scores)
    coverage = coverage_error(targets, scores) - 1
    ranking_loss = label_ranking_loss(targets, scores)
    precision = label_ranking_average_precision_score(targets, scores)
    performance = {'HA': ha, 'one': one, 'Cvg': coverage, 'RANK': ranking_loss, 'Prec': precision}
    return performance

def Yeast_Repro(classifier):
    X, Y = fetch_openml("yeast", version=4, return_X_y=True)
    Y = Y == "TRUE"
    X, Y = X.to_numpy(), Y.to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # X_train = sparse.lil_matrix(X_train)
    # Y_train = sparse.lil_matrix(Y_train)
    # X_test = sparse.lil_matrix(X_test)

    classifier.fit(X_train, Y_train)
    results, scores = classifier.predict(X_test)
    results = results.toarray()
    scores = scores.toarray()
    return results, scores, Y_test

# print("---Yeast Experiment Reproduction---")
# for k in [8, 9, 10, 11, 12]:
#     classifier = MLkNN(k=k)
#     results, scores, Y_true = Yeast_Repro(classifier)
#     print("k =", k, ":", repro_evaluation(Y_true, results, scores))
print("---MLkNN on StackEx---")
x_train = torch.load("C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/train/X.pt")
train_entailments = x_train[:,:, 2]
train_contradictions = x_train[:,:,0]
x_train = train_entailments/(train_entailments+train_contradictions)
y_train = torch.load("C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/train/Y_true.pt")

x_test = torch.load("C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/test/X.pt")
y_test = torch.load("C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/test/Y_true.pt")

test_entailments = x_test[:,:,2]
test_contradictions = x_test[:,:,0]
x_test = test_entailments/(test_entailments+test_contradictions)
y_train = y_train.astype(np.bool)
for k in [3,4,5,6,7,8,9,10,11,12]:
    classifier = MLkNN(k=k)
    classifier.fit(x_train, y_train)

    results, scores = classifier.predict(x_test)
    print("performance for k=", k, ":", evaluation(y_test, results.toarray()))
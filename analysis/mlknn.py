import scipy.sparse
from skmultilearn.adapt import MLkNN
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from scipy import sparse
from pandas import DataFrame
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.metrics import label_ranking_average_precision_score, zero_one_loss, coverage_error, label_ranking_loss
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import xml.etree.ElementTree as etree
from collections import defaultdict
from tqdm import tqdm
import re

CLEANR = re.compile('<.*?>')


def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext


def pullTags(raw_html):
    tags = re.findall(CLEANR, raw_html)
    tags = [tag.replace("<", "").replace(">", "").replace("-", " ") for tag in tags]
    return tags
def stackex_philosophy(data_root):
    data = {}
    qa_pairs = {}
    questions = {}
    answers = {}
    labels = []
    for event, elem in tqdm(etree.iterparse(data_root + "/Posts.xml", events=('end',)), desc="Parsing {} XML file".format("stackexchange_philosophy")):
        if elem.tag == "row":
            attribs = defaultdict(lambda: None, elem.attrib)
            if attribs["PostTypeId"] == '1':
                tags = pullTags(attribs["Tags"])
                for tag in tags:
                    if tag not in labels:
                        labels.append(tag)
                questions[int(attribs["Id"])] = {"Body": cleanhtml(attribs["Body"].replace("\n", " ")), "Tags": tags}
                if "AcceptedAnswerId" in attribs.keys():
                    qa_pairs[int(attribs["Id"])] = int(attribs["AcceptedAnswerId"])
                    if int(attribs["AcceptedAnswerId"]) in answers.keys():
                        questions[int(attribs["id"])]["Body"] += " " + answers[int(attribs["AcceptedAnswerId"])]["Body"]
            if attribs["PostTypeId"] == '2':
                answers[int(attribs["Id"])] = {"Body": cleanhtml(attribs["Body"].replace("\n", " "))}
                if int(attribs["ParentId"]) in qa_pairs.keys() and int(attribs["Id"]) == qa_pairs[int(attribs["ParentId"])]:
                    questions[int(attribs["ParentId"])]["Body"] += " " + answers[int(attribs["Id"])]["Body"]
    for key, value in questions.items():
        data[value["Body"]] = value["Tags"]
    return data, labels
def delete_labels_without_embeddings(embeddings_file, label2id, X, Y):
    word_embeddings = {}
    with open(embeddings_file, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                values = line.split()
                word = str(values[0]).upper()
                word_embeddings[word] = torch.Tensor(np.asarray(values[1:], dtype=np.float32))
            except:
                pass
    label2glove = {}
    #print(word_embeddings)
    no_embeddings = []
    label_embeddings = []
    for label, idx in label2id.items():
        # TODO: replace with pretrained tokenizer
        tokens = label.replace(' ', '<sep>').replace(',', '<sep>').replace('/', '<sep>').replace('-', '<sep>').split('<sep>')

        token_embeddings = []
        for t in tokens:
            t = t.upper()
            if t and t in word_embeddings.keys():
                token_embeddings.append(word_embeddings[t.upper()])
            else:
                #print(t)
                #token_embeddings.append(torch.tensor(100.02))
                #print(label)
                no_embeddings.append(label2id[label])
    label_totals = np.sum(Y, axis=0)
    for i in range(len(label_totals)):
        if label_totals[i] < 3:
            no_embeddings.append(i)
    no_embeddings = np.unique(no_embeddings)
    id2label = {}
    for key, value in label2id.items():
        id2label[value] = key
    no_embeddings = np.sort(no_embeddings)[::-1]
    for label_id in no_embeddings:
        #X = np.delete(X, label_id, 1)
        Y = np.delete(Y, label_id, 1)
        del label2id[id2label[label_id]]
    for i in range(len(X)-1, -1, -1):
        if np.sum(Y[i]) == 0:
            Y = np.delete(Y, i, 0)
            X = np.delete(X, i, 0)
    return X, Y, label2id
def vectorized_stackex(num_samples = 5000):
    data, labels = stackex_philosophy("C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy")
    vectorizor = TfidfVectorizer()
    vectorized = []
    samples = list(data.keys())
    samples = samples[0:num_samples]
    Y = list(data.values())
    Y = Y[0:num_samples]
    label2id = {}
    i = 0
    for label in labels:
        label2id[label] = i
        i += 1
    Y_hot = np.zeros((len(Y), len(labels)))
    for i in range(len(Y)):
        for label in Y[i]:
            Y_hot[i, label2id[label]] = 1
    X, Y, label2id = delete_labels_without_embeddings('../inputs/glove.6B.100d.txt', label2id, samples, Y_hot)
    # for i in range(len(samples)):
    #     vectorized.append(vectorizor.fit_transform(samples[i])).toarray()

    vectorized = vectorizor.fit_transform(X).toarray()



    return vectorized, Y, label2id
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
# x_train = torch.load("C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/train/X.pt")
# train_entailments = x_train[:,:, 2]
# train_contradictions = x_train[:,:,0]
# x_train = train_entailments/(train_entailments+train_contradictions)
# y_train = torch.load("C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/train/Y_true.pt")
#
# x_test = torch.load("C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/test/X.pt")
# y_test = torch.load("C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/test/Y_true.pt")
#
# test_entailments = x_test[:,:,2]
# test_contradictions = x_test[:,:,0]
# x_test = test_entailments/(test_entailments+test_contradictions)
# y_train = y_train.astype(bool)
X, Y, label2id = vectorized_stackex()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
Y_train = Y_train.astype(bool)
for k in [1,2,3,4,5,6,7,8,9,10,11,12]:
    classifier = MLkNN(k=k)
    classifier.fit(X_train, Y_train)
    results, scores = classifier.predict(X_test)
    print("performance for k=", k, ":", evaluation(Y_test, results.toarray()))
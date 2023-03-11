from skmultilearn.adapt import MLARAM
import torch
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset
import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

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
    X, Y, label2id = delete_labels_without_embeddings('../update/inputs/glove.6B.100d.txt', label2id, samples, Y_hot)
    # for i in range(len(samples)):
    #     vectorized.append(vectorizor.fit_transform(samples[i])).toarray()

    vectorized = vectorizor.fit_transform(X).toarray()
    return vectorized, Y, label2id
def Yeast_Repro(classifier):
    X, Y = fetch_openml("yeast", version=4, return_X_y=True)
    Y = Y == "TRUE"
    X, Y = X.to_numpy(), Y.to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # X_train = sparse.lil_matrix(X_train)
    # Y_train = sparse.lil_matrix(Y_train)
    # X_test = sparse.lil_matrix(X_test)

    classifier.fit(X_train, Y_train)
    # results, scores = classifier.predict(X_test)
    # results = results.toarray()
    # scores = scores.toarray()
    # return results, scores, Y_test
    results = classifier.predict(X_test)
    return results, Y_test
def evaluation(targets, predictions):
    acc = accuracy_score(targets, predictions)
    ha = 1 - hamming_loss(targets, predictions)
    ebf1 = f1_score(targets, predictions, average='samples', zero_division=0)
    mif1 = f1_score(targets, predictions, average='micro', zero_division=0)
    maf1 = f1_score(targets, predictions, average='macro', zero_division=0)
    performance = {'ACC': acc, 'HA': ha, 'ebF1': ebf1, 'miF1': mif1, 'maF1': maf1}
    return performance
def eurlex(split = 'train'):
    dataset = load_dataset("eurlex")
    train = dataset[split]
    train = train.to_pandas()
    train = train.to_dict()
    del train['celex_id']
    del train['title']
    removed = []
    X = []
    Y = []
    labels = []
    for i in train['eurovoc_concepts'].keys():
        for label in train['eurovoc_concepts'][i]:
            labels.append(label)
        if len(train['eurovoc_concepts'][i]) == 0:
            removed.append(i)
            del train['eurovoc_concepts'][i]
            del train['text'][i]
    for i in train['text'].keys():
        X.append(train['text'][i])
        Y.append(train['eurovoc_concepts'][i])
    labels = np.unique(labels)
    label2id = {}
    i = 0
    for label in labels:
        label2id[label] = i
        i = i + 1
    return X, Y, label2id
from sklearn.feature_extraction.text import TfidfVectorizer
def vectorized_eurlex(X, Y, label2id):

    vectorizor = TfidfVectorizer()

    Y_hot = np.zeros((len(Y), len(label2id)))
    for i in range(len(Y)):
        for label in Y[i]:
            Y_hot[i, label2id[label]] = 1

    # for i in range(len(samples)):
    #     vectorized.append(vectorizor.fit_transform(samples[i])).toarray()

    vectorized = vectorizor.fit_transform(X)



    return vectorized, Y_hot

X, Y, label2id = vectorized_stackex()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
Y_train = Y_train.astype(bool)
classifier = MLARAM()
classifier.fit(X_train, Y_train)
results = classifier.predict(X_test)
try:
    print(evaluation(Y_test, results))
except:
    print(evaluation(Y_test, results.toarray()))

# classifier = MLARAM()
# results, Y_test = Yeast_Repro(classifier)
# print(evaluation(Y_test, results))
# X, Y, label2id = eurlex()
# vectorized, Y = vectorized_eurlex(X,Y,label2id)
# vectorized = vectorized[0:20000]
# vectorized = sparse.csr_matrix(vectorized)
# Y = Y[0:20000]
# Y.astype(bool)
# classifier.fit(vectorized,Y)
# print("--testing--")
# X, Y, label2id = eurlex()
# vectorized, Y = vectorized_eurlex(X,Y,label2id)
# vectorized = vectorized[20000:40000]
# vectorized = sparse.csr_matrix(vectorized)
#
# Y = Y[20000:40000]
# predictions = classifier.predict(vectorized)
# print(evaluation(Y, predictions))
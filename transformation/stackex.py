import os
import torch
import numpy as np
import random
import readers
from converter import DataConverter


def split_and_save_data(X, Y, path, seed, ratio):
    random.seed(seed)
    choices = list(range(X.shape[0]))
    trainIdx = random.sample(choices, k=int(ratio * X.shape[0]))
    testIdx = list(set(choices) - set(trainIdx))
    x_train = np.array([X[i] for i in trainIdx])
    x_test = np.array([X[i] for i in testIdx])
    y_train = np.array([Y[i] for i in trainIdx])
    y_test = np.array([Y[i] for i in testIdx])
    torch.save(x_train, path + "train/X.pt")
    torch.save(y_train,  path + "train/Y_true.pt")
    torch.save(x_test, path + "test/X.pt")
    torch.save(y_test, path + "test/Y_true.pt")


def remove_labels_without_embeddings(embeddings_file, label2id, X, Y):
    word_embeddings = {}
    with open(embeddings_file, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                values = line.split()
                word = str(values[0]).upper()
                word_embeddings[word] = torch.Tensor(np.asarray(values[1:], dtype=np.float32))
            except:
                pass
    no_embeddings = []
    for label, idx in label2id.items():
        tokens = label.replace(' ', '<sep>').replace(',', '<sep>').replace('/', '<sep>').replace('-', '<sep>').split('<sep>')
        token_embeddings = []
        for t in tokens:
            t = t.upper()
            if t and t in word_embeddings.keys():
                token_embeddings.append(word_embeddings[t.upper()])
            else:
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
        X = np.delete(X, label_id, 1)
        Y = np.delete(Y, label_id, 1)
        del label2id[id2label[label_id]]
    for i in range(len(X)-1, -1, -1):
        if np.sum(Y[i]) == 0:
            Y = np.delete(Y, i, 0)
            X = np.delete(X, i, 0)
    return X, Y, label2id


device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "./inputs/philosophy.stackexchange.com/"
results_root = "./outputs/stackex_philosophy/"
if not os.path.exists(results_root):
    os.makedirs(results_root)
data = readers.stackex_philosophy(data_root)
label2id = {}
i = 0
for label in data[1]:
    label2id[label] = i
    i = i + 1
conversion = DataConverter(data[0], label2id, device=device)
X = conversion.X.cpu().numpy()
Y = conversion.Y.cpu().numpy()

X, Y, label2id = remove_labels_without_embeddings('../update/inputs/glove.6B.100d.txt', label2id, X, Y)
split_and_save_data(X, Y, results_root, 1, 0.8)
torch.save(label2id, results_root + '/label2id.pt')


import torch
import numpy as np
import random

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
        X = np.delete(X, label_id, 1)
        Y = np.delete(Y, label_id, 1)
        del label2id[id2label[label_id]]
    for i in range(len(X)-1, -1, -1):
        if np.sum(Y[i]) == 0:
            Y = np.delete(Y, i, 0)
            X = np.delete(X, i, 0)
    return X, Y, label2id
def split(X, Y, path, seed, ratio):
    random.seed(seed)
    choices = list(range(X.shape[0]))
    trainIdx = random.sample(choices, k=int(ratio * X.shape[0]))
    testIdx = list(set(choices) - set(trainIdx))
    x_train = np.array([X[i] for i in trainIdx])
    x_test = np.array([X[i] for i in testIdx])
    y_train = np.array([Y[i] for i in trainIdx])
    y_test = np.array([Y[i] for i in testIdx])
    torch.save(x_train, path + "train/X.pt")
    torch.save(y_train,  path+ "train/Y_true.pt")
    torch.save(x_test, path+ "test/X.pt")
    torch.save(y_test,path+ "test/Y_true.pt")
def save_similarity(embeddings_file, label2id, path):
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
                no_embeddings.append(label)
        if label in no_embeddings:
            continue
        #label_embeddings = torch.mean(torch.stack(token_embeddings, -1), -1)
        label_embeddings.append(torch.mean(torch.stack(token_embeddings, -1), -1))
        #label2glove[idx] = label_embeddings
    idx = 0
    for embeddings in label_embeddings:
        label2glove[idx] = embeddings
        idx += 1
    #print(no_embeddings)
    for label in no_embeddings:
        del label2id[label]
    T = torch.zeros((len(label2glove), label2glove[0].size(0)))
    for idx, embedding in label2glove.items():
        T[idx, :] = embedding
    norm = torch.linalg.norm(T, dim=1).unsqueeze(1)
    similarity = torch.matmul(T, T.transpose(0, 1)) / torch.matmul(norm, norm.transpose(0, 1))
    similarity = torch.nan_to_num(similarity, nan=similarity.nanmedian())
    similarity = similarity.fill_diagonal_(similarity.nanmedian())
    torch.save(similarity, path + 'similarity.pt' )
if __name__ == '__main__':

    label2id = torch.load('C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/full/label2id.pt')
    X = torch.load("C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/full/X.pt")
    Y = torch.load("C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/full/Y_true.pt")
    X, Y, label2id = delete_labels_without_embeddings('../inputs/glove.6B.100d.txt', label2id, X, Y)
    torch.save(label2id, 'C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/label2id.pt')
    path = "C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/"
    split(X,Y,path, 1, 0.8)
    path = "C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000/"
    save_similarity("../inputs/glove.6B.100d.txt", label2id, path)

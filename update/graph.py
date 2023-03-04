#TODO
import pickle

import torch
import numpy as np

__all__ = ['by_word_embeddings', 'by_hierarchy_tree', 'by_random', 'by_memory']


def by_word_embeddings(embeddings_file, label2id, percentile_neg=0.4, percentile_pos=0.6, **kwargs):
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
    for label, idx in label2id.items():
        # TODO: replace with pretrained tokenizer
        tokens = label.replace(' ', '<sep>').replace(',', '<sep>').replace('/', '<sep>').replace('-', '<sep>').split('<sep>')
        token_embeddings = []
        for t in tokens:
            if t and t in word_embeddings.keys():
                token_embeddings.append(word_embeddings[t])
        label_embeddings = torch.mean(torch.stack(token_embeddings, -1), -1)
        label2glove[idx] = label_embeddings
    T = torch.zeros((len(label2glove), label2glove[0].size(0)))
    for idx, embedding in label2glove.items():
        T[idx, :] = embedding
    norm = torch.linalg.norm(T, dim=1).unsqueeze(1)
    similarity = torch.matmul(T, T.transpose(0, 1)) / torch.matmul(norm, norm.transpose(0, 1))
    similarity = torch.nan_to_num(similarity, nan=similarity.nanmedian())
    similarity = similarity.fill_diagonal_(similarity.nanmedian())
    lower = torch.quantile(similarity, percentile_neg)
    upper = torch.quantile(similarity, percentile_pos)
    adj = 1 * (similarity >= upper) - 1 * (similarity <= lower)
    return adj


def by_hierarchy_tree(hierarchy_file, label2id, **kwargs):
    adj = torch.zeros((len(label2id.values()), len(label2id.values())), dtype = torch.int32, device = torch.device('cuda:0'))
    with open(hierarchy_file, "rb") as f:
        hierarchy = pickle.load(f)
    for parent in hierarchy.keys():
        for child in hierarchy[parent]:
            for sibling in hierarchy[parent]:
                adj[label2id[child], label2id[sibling]] = -1
            if parent in label2id.values():
                adj[label2id[child], label2id[parent]] = 1
    for i in range(len(label2id.values())):
        adj[i,i] = 0
    return adj


def by_random(label2id, **kwargs):
    num_labels = len(label2id)
    adj = torch.zeros((num_labels, num_labels))  # in {-1, 0, 1}, diagonal zero, undirected
    vals = torch.randint(-1, 2, (1, int(num_labels*(num_labels+1)/2)))
    i, j = torch.triu_indices(num_labels, num_labels)
    adj[i, j] = vals.float()
    adj.T[i, j] = vals.float()
    return adj


def by_memory(similarity_file="./update/inputs/reuters/similarity.pt",
              percentile_neg=0.4, percentile_pos=0.6, **kwargs):
    similarity = torch.load(similarity_file)
    lower = torch.quantile(similarity, percentile_neg)
    upper = torch.quantile(similarity, percentile_pos)
    adj = 1 * (similarity >= upper) - 1 * (similarity <= lower)
    return adj
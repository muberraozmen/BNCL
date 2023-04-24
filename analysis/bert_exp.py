import torch
import numpy as np
percentile_neg=  0.1
percentile_pos  = 0.9
bert_similarity = torch.load("bert_similarity.pt")
lower = torch.quantile(bert_similarity, percentile_neg)
upper = torch.quantile(bert_similarity, percentile_pos)
bert_adj = 1 * (bert_similarity >= upper) - 1 * (bert_similarity <= lower)

similarity = torch.load("similarity.pt")
lower = torch.quantile(similarity, percentile_neg)
upper = torch.quantile(similarity, percentile_pos)
glove_adj = 1 * (similarity >= upper) - 1 * (similarity <= lower)

label2id = torch.load("reuters_label2id.pt")
id2label = {}

for label, id in label2id.items():
    id2label[id] = label
count = 0
n_e_bert = 0
n_e_glove = 0
print(len(glove_adj))
for i in range(len(glove_adj)):
    for j in range(i,len(glove_adj)):
        n_e_bert += (bert_adj[i,j])
        n_e_glove += (glove_adj[i,j])
        if (bert_adj[i,j] > 0) and (glove_adj[i,j] < 0):
            print(id2label[i], ",", id2label[j], np.round(float(bert_similarity[i,j]), decimals=4 ), np.round(float(similarity[i,j]), decimals=4))
            count+= 1
        elif (bert_adj[i,j] < 0) and (glove_adj[i,j] > 0):
            print(id2label[i], ",", id2label[j], np.round(float(bert_similarity[i,j]), decimals=4 ), np.round(float(similarity[i,j]), decimals=4))
            count += 1
        # elif (bert_adj[i,j] == 0) and (glove_adj[i,j] != 0):
        #     print(id2label[i], ",", id2label[j], bert_similarity[i, j], similarity[i, j])
        #     count += 1
        # elif (bert_adj[i,j] != 0) and (glove_adj[i,j] == 0):
        #     print(id2label[i], ",", id2label[j], bert_similarity[i, j], similarity[i, j])
        #     count += 1
print(count)
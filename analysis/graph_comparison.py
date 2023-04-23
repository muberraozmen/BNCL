import torch
from update.graph import *
import matplotlib.pyplot as plt
import networkx

data_dir = '/Users/mob/Documents/PycharmProjects/BNCL/update/inputs/reuters'
label2id = torch.load(data_dir + '/' + 'label2id.pt')
glove_file = data_dir + '/' + 'similarity.pt'
bert_file = data_dir + '/' + 'bert_similarity.pt'

# adj_glove = by_word_embeddings(embeddings_file, label2id, percentile_neg=0.2, percentile_pos=0.8)
adj_glove = by_memory(glove_file, percentile_neg=0.1, percentile_pos=0.9)
adj_bert = by_memory(bert_file, percentile_neg=0.1, percentile_pos=0.9)

fig, axs = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
plt.subplot(1, 2, 1)
plt.imshow(adj_glove, vmin=-1, vmax=1, cmap='RdYlGn', aspect='auto')
axs[0].set_title("Glove")
plt.subplot(1, 2, 2)
plt.imshow(adj_bert, vmin=-1, vmax=1, cmap='RdYlGn', aspect='auto')
axs[1].set_title("Bert")
plt.colorbar()
# plt.savefig('bertvsglove.pdf')
# plt.show()
print("Total number of positive edges {:}".format((adj_glove == 1).sum()))
print("Total number of negative edges {:}".format((adj_glove == -1).sum()))
print("Number of shared positive edges {:}".format(((adj_bert == 1) * (adj_glove == 1)).sum()))
print("Number of shared negative edges {:}".format(((adj_bert == -1) * (adj_glove == -1)).sum()))
print("Number of conflicts {:}".format(((adj_bert == 1) * (adj_glove == -1)).sum() + ((adj_bert == -1) * (adj_glove == 1)).sum()))

glove = networkx.from_numpy_matrix(adj_glove.numpy())
bert = networkx.from_numpy_matrix(adj_bert.numpy())
# print(networkx.eigenvector_centrality(glove))
# print(networkx.eigenvector_centrality(bert))
print(networkx.average_clustering(glove))
print(networkx.average_clustering(bert))


print()



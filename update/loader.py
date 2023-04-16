import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from graph import *

__all__ = ['DataLoader', 'TransformedData']


class TransformedData(Dataset):
    def __init__(self, data_dir, supervision, **kwargs):
        self.data_dir = data_dir
        self.supervision = supervision
        self.X = torch.softmax(torch.as_tensor(torch.load(data_dir + '/train/' + 'X.pt')), dim=2)
        self.Y_true = torch.as_tensor(torch.load(data_dir + '/train/' + 'Y_true.pt'))
        self.label2id = torch.load(data_dir + '/' + 'label2id.pt')
        self.num_samples = self.Y_true.shape[0]
        self.num_labels = self.Y_true.shape[1]
        self.kappa = None
        self.lambdas = None
        self.adj = None
        self.annotated_idx = None
        if supervision == 1:
            self.annotation_free(**kwargs)
        elif supervision == 2:
            self.scarce_annotation(**kwargs)
        elif supervision == 3:
            self.domain_supervisor(**kwargs)
        else:
            raise NotImplementedError

        if self.annotated_idx is not None:
            self.annotated_data = self.X[self.annotated_idx, :, :]
            self.annotated_target = self.Y_true[self.annotated_idx, :]
            rest_idx = torch.arange(0, self.num_samples)
            rest_idx = rest_idx[~rest_idx.unsqueeze(-1).eq(self.annotated_idx).any(1)]
            self.X = self.X[rest_idx, :, :]
            self.Y_true = self.Y_true[rest_idx, :]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        entailment = self.X[i, :, 2]
        contradiction = self.X[i, :, 0]
        return entailment, contradiction

    def annotation_free(self, embeddings_file, **kwargs):
        self.kappa = self.Y_true.mean()
        self.lambdas = self.Y_true.mean(0)
        # self.adj = by_memory(similarity_file=self.data_dir + "/similarity.pt", **kwargs)
        self.adj = by_word_embeddings(embeddings_file, label2id=self.label2id, **kwargs)

    def scarce_annotation(self, embeddings_file=None, annotation_ratio=10, **kwargs):
        print(self.num_samples*annotation_ratio//100)
        self.annotated_idx = torch.randint(0, self.num_samples, (self.num_samples*annotation_ratio//100, )).long()
        self.kappa = self.Y_true[self.annotated_idx, :].mean()
        self.lambdas = self.Y_true[self.annotated_idx, :].mean(0)
        # self.adj = by_memory(similarity_file=self.data_dir + "/similarity.pt", **kwargs)
        self.adj = by_word_embeddings(embeddings_file, label2id=self.label2id, **kwargs)

    def domain_supervisor(self, hierarchy_file=None, annotation_ratio=10, **kwargs):
        self.annotated_idx = torch.randint(0, self.num_samples, (self.num_samples*annotation_ratio//100, )).long()
        self.kappa = self.Y_true.mean()
        self.lambdas = self.Y_true.mean(0)
        # self.adj = by_memory(similarity_file=self.data_dir + "/similarity.pt", **kwargs)
        self.adj = by_hierarchy_tree(hierarchy_file, label2id=self.label2id, **kwargs)

    def load_test(self):
        data = torch.softmax(torch.as_tensor(torch.load(self.data_dir + '/test/' + 'X.pt')), dim=2)
        targets = torch.as_tensor(torch.load(self.data_dir + '/test/' + 'Y_true.pt'))
        entailments = data[:, :, 2]
        contradictions = data[:, :, 0]
        return entailments, contradictions, targets

    def load_annotated(self):
        return AnnotatedData(self.annotated_data, self.annotated_target)


class AnnotatedData(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.Y_true = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        entailment = self.X[i, :, 2]
        contradiction = self.X[i, :, 0]
        target = self.Y_true[i, :]
        return entailment, contradiction, target


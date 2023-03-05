import numpy as np
import random
import os
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from scipy.sparse import csr_matrix, vstack


def merge_original(data_root):
    train_features, train_labels = read_data(data_file=data_root + '/train.txt')
    valid_features, valid_labels = read_data(data_file=data_root + '/valid.txt')
    test_features, test_labels = read_data(data_file=data_root + '/test.txt')
    features = vstack([train_features, valid_features, test_features])
    labels = vstack([train_labels, valid_labels, test_labels])
    return features, labels


def split_random(features, labels):
    num_samples, num_features = features.shape[0], features.shape[1]
    _, num_labels = labels.shape[0], labels.shape[1]
    idx = list(range(num_samples))
    test_idx = random.sample(idx, int(num_samples*0.1))
    valid_idx = random.sample(set(idx) - set(test_idx), int(num_samples*0.1))
    train_idx = list(set(idx) - set(valid_idx) - set(test_idx))
    train_features, train_labels = features[train_idx], labels[train_idx]
    valid_features, valid_labels = features[valid_idx], labels[valid_idx]
    test_features, test_labels = features[test_idx], labels[test_idx]
    return (train_features, train_labels), (valid_features, valid_labels), (test_features, test_labels)


def read_data(data_file):
    def seq2bin(seq, shape):
        indices = []
        indptr = [0]
        offset = 0
        for item in seq:
            if len(item) > 0:
                indices.extend(item)
                offset += len(item)
            indptr.append(offset)
        return csr_matrix((np.array([1.0] * len(indices)), np.array(indices), np.array(indptr)), shape=shape)
    with open(data_file, 'rb') as f:
        # skip the headers and read the data statistics
        f.readline().decode('utf-8').rstrip("\n")
        f.readline().decode('utf-8').rstrip("\n")
        f.readline().decode('utf-8').rstrip("\n")
        line = f.readline().decode('utf-8').rstrip("\n")
        line = line.split(" ")
        num_samples, num_features, num_labels = int(line[1]), int(line[2]), int(line[3])
        # load features and labels to csr matrix
        features, labels = load_svmlight_file(f, n_features=num_features, multilabel=True)
        labels = seq2bin(labels, shape=(num_samples, num_labels))
    return features, labels


def write_split(features, labels, data_file):
    num_samples, num_features = features.shape[0], features.shape[1]
    _, num_labels = labels.shape[0], labels.shape[1]
    comment = "{:} {:} {:}".format(num_samples, num_features, num_labels)
    dump_svmlight_file(features, labels, data_file, multilabel=True, comment=comment)


data_root = '/home/muberra/Desktop/PycharmProjects/RGMP/data/bibtext/'
features, labels = merge_original(data_root=data_root)
for k in range(10):
    split_dir = data_root + '/' + str(k+1) + '/'
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    (trainX, trainY), (validX, validY), (testX, testY) = split_random(features, labels)
    write_split(trainX, trainY, split_dir + '/train.txt')
    write_split(validX, validY, split_dir + '/valid.txt')
    write_split(testX, testY, split_dir + '/test.txt')

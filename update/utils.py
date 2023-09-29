import argparse
import os
import time
import json
import logging
import logging.config
import sys
import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss, f1_score

def cluster(lambdas, n, start=0.3, stop=0.0001):
    k_range = np.linspace(0.001, start, num=n-1)
    k_range = np.concatenate([[0], k_range, [1]])
    clusters = np.zeros(len(lambdas))
    means = np.zeros(n)
    counts = np.zeros(n)
    for i in range(len(lambdas)):
        for k in range(len(k_range)-1):
            if lambdas[i] >= k_range[k] and lambdas[i] < k_range[k+1]:
                clusters[i] = k
                means[k] += lambdas[i]
                counts[k] += 1
    for i in range(len(means)):
        if counts[i] != 0:
            means[i] = means[i]/counts[i]
    return clusters, means, counts, k_range[1:-1]
def get_logger(results_dir, name='log'):
    config_dict = {"version": 1,
                   "formatters": {"base": {"format": "%(message)s"}},
                   "handlers": {"base": {"class": "logging.FileHandler",
                                         "level": "DEBUG",
                                         "formatter": "base",
                                         "filename": results_dir + name,
                                         "encoding": "utf8"}},
                   "root": {"level": "DEBUG", "handlers": ["base"]},
                   "disable_existing_loggers": False}
    logging.config.dictConfig(config_dict)

    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger


def set_seed(seed, multiple_device=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if multiple_device:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluation(targets, predictions):
    acc = accuracy_score(targets, predictions)
    ha = 1 - hamming_loss(targets, predictions)
    ebf1 = f1_score(targets, predictions, average='samples', zero_division=0)
    mif1 = f1_score(targets, predictions, average='micro', zero_division=0)
    maf1 = f1_score(targets, predictions, average='macro', zero_division=0)
    performance = {'ACC': acc, 'HA': ha, 'ebF1': ebf1, 'miF1': mif1, 'maF1': maf1}
    return performance


def ensemble_evaluation(targets, predictions):

    acc = accuracy_score(targets, predictions)
    ha = 1 - hamming_loss(targets, predictions)
    ebf1 = f1_score(targets, predictions, average='samples', zero_division=0)
    mif1 = f1_score(targets, predictions, average='micro', zero_division=0)
    maf1 = f1_score(targets, predictions, average='macro', zero_division=0)
    performance = {'ACC': acc, 'HA': ha, 'ebF1': ebf1, 'miF1': mif1, 'maF1': maf1}
    return performance


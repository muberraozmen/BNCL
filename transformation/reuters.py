import os
import torch
import readers
from converter import DataConverter

device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = './inputs/reuters21578/'
results_root = "./outputs/reuters21578/"
if not os.path.exists(results_root):
    os.makedirs(results_root)
data = readers.reuters(data_root)
label2id = {}
i = 0
for label in data["topics_vocab"].values():
    label2id[label.upper()] = i
    i = i + 1
torch.save(label2id, results_root + '/label2id.pt')
DataConverter(data["samples"]["train"], label2id, device=device, data_dir=results_root + "/train/")
DataConverter(data["samples"]["test"], label2id, device=device, data_dir=results_root + "/test/")

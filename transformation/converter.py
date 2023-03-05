import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import os
import readers


class DataConverter(object):
    def __init__(self, samples, label2id, device, data_dir):
        self.num_samples = len(samples)
        self.num_labels = len(label2id)
        self.label2id = label2id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
        model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
        self.model = model.to(self.device)
        X = []
        Y = []
        i = 0
        j = 0
        print(label2id)
        self.id2hypothesis = self.get_hypotheses()
        for src, tgt in samples.items():
            premise = self.tokenize_premise(src)
            try:
                if j%200 == 0:
                    print("j is:", j)
                tic = time.perf_counter()
                X.append(self.calculate_odds(premise))
                Y.append(self.onehot(tgt))
                toc = time.perf_counter()
                j = j + 1
                #print(f"One sample processing time {toc - tic:0.4f} seconds")
            except:
                i += 1
                if i == 1:
                    print('PROBLEM \n')
                    print(src, '\n')
                    print(tgt, '\n')
                    print(self.onehot(tgt), '\n')
                    print("i: ", i, '\n')
                    print("j: ", j, '\n')
                pass
        self.X = torch.stack(X, dim=0)
        self.Y = torch.stack(Y, dim=0)
        self.Y_pred = self.get_independent_estimations()
        self.save_data(data_dir)

    def tokenize_premise(self, premise):
        """ https://huggingface.co/docs/transformers/pad_truncation """
        return self.tokenizer.encode(premise, return_tensors='pt', truncation=True, max_length=128)

    def tokenize_hypothesis(self, label):
        hypothesis = f'This is about {label}.'
        seq = self.tokenizer.encode(hypothesis, return_tensors='pt', truncation=False)
        seq[:, 0] = self.tokenizer.eos_token_id
        return seq

    def get_hypotheses(self):
        id2hypothesis = {}
        for label, idx in self.label2id.items():
            id2hypothesis[idx] = self.tokenize_hypothesis(label)
        return id2hypothesis

    def form_input(self, premise):
        inputs = {}
        for idx, hypothesis in self.id2hypothesis.items():
            inputs[idx] = torch.cat((premise, hypothesis), dim=1)
        # max_length = max([i.squeeze().numel() for i in inputs])
        # inputs = [torch.nn.functional.pad(i, pad=(0, max_length - i.numel()), mode='constant', value=self.tokenizer.pad_token_id) for i in inputs]
        # torch.stack(inputs, dim=0).squeeze()
        return inputs

    def onehot(self, tgt):
        y = torch.zeros(self.num_labels)
        idx = [self.label2id[l] for l in tgt]
        y[idx] = 1
        return y

    def calculate_odds(self, premise):
        inputs = self.form_input(premise)
        odds = torch.zeros(self.num_labels, 3)
        for idx, seq in inputs.items():
            odds[idx] = self.model(seq.to(self.device)).logits.cpu().clone().detach()
        return odds

    def get_independent_estimations(self):
        p = []
        for sample in self.X:
            p.append(sample[:, [0, 2]].softmax(dim=1)[:, 1])
        return torch.stack(p, dim=0)

    def save_data(self, data_dir):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        torch.save(self.X.cpu().numpy(), data_dir + '/X.pt')
        torch.save(self.Y.cpu().numpy(), data_dir + '/Y_true.pt')
        torch.save(self.Y_pred.cpu().numpy(), data_dir + '/Y_pred0.pt')

#
# data_root = "C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/RCV1"
# reduce_data = False
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# data = readers.RCV1(data_root, sample=False)
# test = data['test']
# label2id = data['label2id']
# torch.save(label2id, data_root + '/label2id.pt')
# if "prior" in data.keys():
#     torch.save(data['prior'], data_root + '/prior.pt')
# print("doing train")
# DataConverter(data['train'], label2id, device=device, data_dir=data_root + "/train/")
# print("doing test")
# DataConverter(data['test'], label2id, device=device, data_dir=data_root + "/test/")
import random
device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "/home/muberra/projects/def-coama74/muberra/XMTC/resources/reuters21578/"
Reuters = readers.Reuters(data_root)
samples = Reuters["samples"]
# rcv1 = readers.RCV1("C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/RCV1", sample=True)
# label2idRCV1 = rcv1['label2id']
# test = rcv1['test']
# reduced = random.sample(list(samples["train"].items()), 2000)
# train_idx = random.sample(range(2000), 1000)
# test_idx = set(range(2000)) - set(train_idx)
# reduced_train = {reduced[idx][0]: reduced[idx][1] for idx in train_idx}
# reduced_test = {reduced[idx][0]: reduced[idx][1] for idx in test_idx}
# print("doing train")
# DataConverter(reduced_train, label2id, device=device, data_dir=data_root + "/train/")
# print("doing test")
# DataConverter(reduced_test, label2id, device=device, data_dir=data_root + "/test/")
labels = Reuters["topics_vocab"].values()
i = 0
label2id = {}
for label in labels:
    label2id[label.upper()] = i
    i = i + 1

torch.save(label2id, data_root + '/label2id.pt')
print("doing train")
DataConverter(samples["train"], label2id, device=device, data_dir=data_root + "full/train/")
print("doing test")
DataConverter(samples["test"], label2id, device=device, data_dir=data_root + "full/test/")
print()

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os


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
        counter = 0
        self.id2hypothesis = self.get_hypotheses()
        for src, tgt in samples.items():
            premise = self.tokenize_premise(src)
            try:
                X.append(self.calculate_odds(premise))
                Y.append(self.onehot(tgt))
                counter = counter + 1
                print(counter)
                if counter % 100 == 0:
                    self.X = torch.stack(X, dim=0)
                    self.Y = torch.stack(Y, dim=0)
                    self.save_data(data_dir + '/first' + str(counter) + '/')
            except:
                pass
        self.X = torch.stack(X, dim=0)
        self.Y = torch.stack(Y, dim=0)
        if data_dir is not None:
            self.save_data(data_dir)

    def tokenize_premise(self, premise):
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


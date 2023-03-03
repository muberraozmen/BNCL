import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['UpdateModel']


class UpdateModel(nn.Module):
    def __init__(self, num_labels, num_layers, adj):
        super(UpdateModel, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(num_layers):
            d_pos, d_neg = self.balanced_neighborhoods(adj, hob=k)
            self.layers.append(Layer(num_labels, d_pos, d_neg))
        self.reset_parameters()

    def forward(self, entailment, contradiction):
        for layer in self.layers:
            entailment,  contradiction = layer(entailment, contradiction)
        return entailment, contradiction

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    @staticmethod
    def balanced_neighborhoods(adj, hob):
        adj = adj.fill_diagonal_(0)
        adj_pos = 1 * (adj > 0)
        adj_neg = 1 * (adj < 0)
        k = 0
        friends, enemies = adj_pos, adj_neg
        friends = friends.type(torch.int32).to(torch.device('cpu'))
        enemies = enemies.type(torch.int32).to(torch.device('cpu'))
        adj_pos = adj_pos.type(torch.int32).to(torch.device('cpu'))
        adj_neg = adj_neg.type(torch.int32).to(torch.device('cpu'))

        while k < hob:
            friends_new = (torch.matmul(adj_pos, friends) + torch.matmul(adj_neg, enemies))
            enemies_new = (torch.matmul(adj_pos, enemies) + torch.matmul(adj_neg, friends))
            friends = torch.where(friends_new >= 1, 1, friends_new).fill_diagonal_(0)
            enemies = torch.where(enemies_new >= 1, 1, enemies_new).fill_diagonal_(0)
            k += 1
        return friends.to(torch.device('cuda:0')), enemies.to(torch.device('cuda:0'))

    def predict(self, entailments, contradictions):
        predictions = 1 * (entailments >= contradictions)
        return predictions


class Layer(nn.Module):
    def __init__(self, num_labels, d_pos, d_neg):
        super(Layer, self).__init__()
        self.w_e_pos = MaskedLinear(num_labels, num_labels, bias=False, mask=d_pos)
        self.w_e_neg = MaskedLinear(num_labels, num_labels, bias=False, mask=d_neg)
        self.w_c_pos = MaskedLinear(num_labels, num_labels, bias=False, mask=d_pos)
        self.w_c_neg = MaskedLinear(num_labels, num_labels, bias=False, mask=d_neg)

    def forward(self, entailment, contradiction):
        entailment_updated = entailment + F.relu(self.w_e_pos(entailment)) + F.relu(self.w_c_neg(contradiction))
        contradiction_updated = contradiction + F.relu(self.w_e_neg(entailment)) + F.relu(self.w_c_pos(contradiction))
        return entailment_updated, contradiction_updated


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, mask=None):
        super().__init__(in_features, out_features, bias)
        if mask is not None:
            self.register_buffer('mask', mask)
        else:
            self.register_buffer('mask', torch.ones(out_features, in_features))

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)






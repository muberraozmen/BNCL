import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['CollectiveLoss']


class CollectiveLoss(object):
    def __init__(self, kappa, lambdas, alphas=None, beta=None):
        self.kappa = torch.as_tensor(kappa)
        self.lambdas = torch.as_tensor(lambdas)
        if alphas is None:
            self.alphas = torch.ones(5)
        else:
            self.alphas = torch.as_tensor(alphas)
        if beta is None:
            self.beta = torch.tensor(10)
        else:
            self.beta = torch.as_tensor(beta)
        self.L1 = nn.MSELoss(reduction='sum')
        self.L2 = nn.MSELoss(reduction='sum')
        self.L3 = nn.MSELoss(reduction='sum')
        self.L4 = nn.MSELoss(reduction='sum')
        self.L5 = nn.BCELoss(reduction='sum')

    def calculate(self, entailment, contradiction, y_true=None):
        l1 = self.L1(entailment + contradiction, torch.ones_like(entailment))
        l2 = - 1 * self.L2(entailment - contradiction, torch.zeros_like(entailment))
        y_pred = 1 / (1 + torch.exp(-self.beta*(entailment - contradiction)))
        l3 = self.L3(y_pred.sum(dim=0), entailment.size(0) * self.lambdas)
        l4 = self.L4(y_pred.sum(dim=1), self.kappa * torch.ones(entailment.size(0)))
        l2 = None
        if l2 is None:
            if y_true is None:
                return self.alphas[0] * l1 + self.alphas[2] * l3 + self.alphas[3] * l4
            else:
                l5 = self.L5(y_pred, y_true)
                return self.alphas[0] * l1 + self.alphas[2] * l3 + self.alphas[3] * l4 + self.alphas[4] * l5

        if y_true is None:
            return self.alphas[0] * l1 + self.alphas[1] * l2 + self.alphas[2] * l3 + self.alphas[3] * l4
        else:
            l5 = self.L5(y_pred, y_true)
            return self.alphas[0] * l1 + self.alphas[1] * l2 + self.alphas[2] * l3 + self.alphas[3] * l4 + self.alphas[4] * l5



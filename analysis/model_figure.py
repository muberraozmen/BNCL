import torch
import numpy as np

y = torch.as_tensor([1, 1, 1, 0, 0])
A = torch.as_tensor([0.2, 0.6, 0.2])
B = torch.as_tensor([0.7, 0.1, 0.2])
C = torch.as_tensor([0.1, 0.8, 0.1])
D = torch.as_tensor([0.1, 0.5, 0.4])
E = torch.as_tensor([0.1, 0.8, 0.1])
F = torch.as_tensor([0.1, 0.5, 0.4])
adj = torch.as_tensor([[0, 0, 0, 0, -1, 1],
                       [0, 0, 0, 0, -1, 0],
                       [0, 0, 0, -1, 0, 0],
                       [0, 0, -1, 0, 1, 0],
                       [-1, -1, 0, 1, 0, 0],
                       [1, 0, 0, 0, 0, 0]])

adj_pos = (1 * (adj > 0)).float()
adj_neg = (1 * (adj < 0)).float()
k = 0
friends_1, enemies_1 = adj_pos, adj_neg
friends_new = (torch.matmul(adj_pos, friends_1) + torch.matmul(adj_neg, enemies_1)).fill_diagonal_(0)
enemies_new = (torch.matmul(adj_pos, enemies_1) + torch.matmul(adj_neg, friends_1)).fill_diagonal_(0)
friends_2 = (1 * (friends_new >= 1)).fill_diagonal_(0)
enemies_2 = (1 * (enemies_new >= 1)).fill_diagonal_(0)

# w_pos_ent_1 = friends_1 * torch.randint(low=1, high=5, size=(6, 6))/10
# w_neg_ent_1 = enemies_1 * torch.randint(low=1, high=5, size=(6, 6))/10
# w_pos_con_1 = friends_1 * torch.randint(low=1, high=5, size=(6, 6))/10
# w_neg_con_1 = enemies_1 * torch.randint(low=1, high=5, size=(6, 6))/10
# w_pos_ent_2 = friends_2 * torch.randint(low=1, high=5, size=(6, 6))/10
# w_neg_ent_2 = enemies_2 * torch.randint(low=1, high=5, size=(6, 6))/10
# w_pos_con_2 = friends_2 * torch.randint(low=1, high=5, size=(6, 6))/10
# w_neg_con_2 = enemies_2 * torch.randint(low=1, high=5, size=(6, 6))/10

w_pos_ent_1 = torch.as_tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.1000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.1000, 0.0000, 0.0000],
                                [0.2000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
w_neg_ent_1 = torch.as_tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.1000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.3000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.2000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.2000, 0.0000, 0.0000, 0.0000],
                                [0.1000, 0.3000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
w_pos_con_1 = torch.as_tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.3000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.3000, 0.0000, 0.0000],
                                [0.1000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
w_neg_con_1 = torch.as_tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.1000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.1000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.1000, 0.0000, 0.0000, 0.0000],
                                [0.2000, 0.1000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
w_pos_ent_2 = torch.as_tensor([[0.0000, 0.3000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.3000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
w_neg_ent_2 = torch.as_tensor([[0.0000, 0.0000, 0.0000, 0.4000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.3000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.1000, 0.0000],
                                [0.4000, 0.3000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.1000, 0.0000, 0.0000, 0.2000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.0000]])
w_pos_con_2 = torch.as_tensor([[0.0000, 0.3000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.3000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
w_neg_con_2 = torch.as_tensor([[0.0000, 0.0000, 0.0000, 0.4000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.1000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.3000, 0.0000],
                                [0.4000, 0.1000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.3000, 0.0000, 0.0000, 0.4000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.4000, 0.0000]])

p_0 = torch.stack((A, B, C, D, E, F), dim=1).transpose(0, 1)
entailment_0 = p_0[:, 0]
contradiction_0 = p_0[:, 2]
entailment_1 = entailment_0 + torch.matmul(w_pos_ent_1, entailment_0) + torch.matmul(w_neg_con_1, contradiction_0)
contradiction_1 = contradiction_0 + torch.matmul(w_neg_ent_1, entailment_0) + torch.matmul(w_pos_con_1, contradiction_0)
entailment_2 = entailment_1 + torch.matmul(w_pos_ent_2, entailment_1) + torch.matmul(w_neg_con_2, contradiction_1)
contradiction_2 = contradiction_1 + torch.matmul(w_neg_ent_2, entailment_1) + torch.matmul(w_pos_con_2, contradiction_1)

print(torch.stack((entailment_0, 1-entailment_0-contradiction_0, contradiction_0), dim=1))
print(torch.stack((entailment_1, 1-entailment_1-contradiction_1, contradiction_1), dim=1))
print(torch.stack((entailment_2, 1-entailment_2-contradiction_2, contradiction_2), dim=1))
print(entailment_2 > contradiction_2)
print()



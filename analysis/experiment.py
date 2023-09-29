import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
label2id = torch.load('C:/Users/jcotn/PycharmProjects/BNCL/resources/stackexchange_philosophy/clean_inputs_data/stackex_philosophy/label2id.pt')
X = torch.load("C:/Users/jcotn/PycharmProjects/BNCL/resources/stackexchange_philosophy/clean_inputs_data/stackex_philosophy/train/X.pt")
Y = torch.load("C:/Users/jcotn/PycharmProjects/BNCL/resources/stackexchange_philosophy/clean_inputs_data/stackex_philosophy/train/Y_true.pt")
X = torch.asarray(X)
X_sm = F.softmax(X, dim=2)
X_sm = np.asarray(X_sm)
n_labels = Y.shape[1]
n_data = X.shape[0]
obs_l = np.sum(Y, axis=0)/n_data
prob_y = np.where(X_sm[:,:,2] > X_sm[:,:,0], 1, 0)
prob_y = np.where(X_sm[:, :, 1] < 0.3, prob_y, 0)
prob_l = np.sum(prob_y, axis=0)/n_data
avg_neutral = np.sum(X_sm[:,:,1], axis=0)/n_data
# plt.figure()
s = 3
# plt.scatter(np.arange(n_labels), obs_l, c='b',s=s )
# plt.scatter(np.arange(n_labels), prob_l, c='r',s=s)
# # plt.scatter(np.arange(n_labels), avg_neutral, c='y',s=s)
# plt.legend(['True', 'E>C', 'Average Neutral'])
# plt.show()

percentiles = np.percentile(obs_l, list(range(0,110, 10)))
sampled_labels = []

for p in range(10):
    # sampled_labels.append([])
    for label_i in range(len(obs_l)):
        if percentiles[p] < obs_l[label_i] < percentiles[p + 1]:
            sampled_labels.append(label_i)
            break
        # sampled_labels[-1] = 'b'
n_dist_sampled_labels = []
for label_i in sampled_labels:
    n_dist_sampled_labels.append(X_sm[:, label_i, 1])

# for label_i in range(len(sampled_labels)):
#     plt.hist( n_dist_sampled_labels[label_i], np.arange(0, 1, 0.01))
#     plt.scatter(np.arange(n_labels), avg_neutral, c='y',s=s)
#     plt.title(percentiles[label_i+1])
#     plt.show()


# import dirichlet
lambdas = torch.load('C:/Users/jcotn/PycharmProjects/BNCL/' + "/update/inputs/stackexchange_philosophy" + '/' + 'lambdas.pt')
lambdas = np.stack(lambdas, axis=0)
print(lambdas[:, 5].shape)
# np.random.shuffle(lambdas[:, 5])
failures = 0
for i in range(lambdas.shape[1]):
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)

    plt.scatter(range(10001), lambdas[:,i])

    plt.axhline(y=obs_l[i], color='r', linewidth=4)

    plt.xlabel("Metropolis-Hastings Iteration Number", fontsize=30)
    plt.ylabel("Sampled Value", fontsize = 30)
    plt.show()
    # if np.abs(obs_l[i] - lambdas[0, i]) < np.abs(obs_l[i] - lambdas[-1, i]):
    #     failures += 1
# print('failures:', failures)
# print('successes:', lambdas.shape[1]-failures)
# dir_params = []
# print(lambdas[:,5])
#
# for i in range(n_labels):
#     print(i)
#     try:
#         dir_params.append(dirichlet.mle(X_sm[:, i, :]))
#     except:
#         print(i, "didnt converge")
#         dir_params.append([0, 0, 0])
# dir_params = np.asarray(dir_params)
# plt.figure()
# plt.show()
# from sklearn import tree
# from sklearn.tree import export_text
# import graphviz
#
# clf = tree.DecisionTreeRegressor(max_depth=3)
# clf = clf.fit(dir_params, obs_l)
# r = export_text(clf, feature_names=['contradiction', 'neutral', 'entailment'])
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# import time
# graph.render("3_layer_d_tree" + str(time.time()))
# print(r)
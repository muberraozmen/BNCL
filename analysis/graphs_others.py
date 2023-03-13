def by_hierarchy_tree(hierarchy_file, label2id, **kwargs):
    adj = torch.zeros((len(label2id.values()), len(label2id.values())), dtype = torch.int32, device = torch.device('cuda:0'))
    with open(hierarchy_file, "rb") as f:
        hierarchy = pickle.load(f)
    for parent in hierarchy.keys():
        for child in hierarchy[parent]:
            for sibling in hierarchy[parent]:
                adj[label2id[child], label2id[sibling]] = -1
            if parent in label2id.values():
                adj[label2id[child], label2id[parent]] = 1
    for i in range(len(label2id.values())):
        adj[i,i] = 0
    return adj


def by_random(label2id, **kwargs):
    num_labels = len(label2id)
    adj = torch.zeros((num_labels, num_labels))  # in {-1, 0, 1}, diagonal zero, undirected
    vals = torch.randint(-1, 2, (1, int(num_labels*(num_labels+1)/2)))
    i, j = torch.triu_indices(num_labels, num_labels)
    adj[i, j] = vals.float()
    adj.T[i, j] = vals.float()
    return adj

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

glove_dir = "/Users/mob/Documents/PycharmProjects/BNCL/update/outputs/reuters/GloVe_Bootstraps/"
bert_dir = "/Users/mob/Documents/PycharmProjects/BNCL/update/outputs/reuters/Bert_Bootstraps/"
setting_dirs = ["supervision level 1/", "supervision level 2/", "supervision level 3/"]
settings_name_map = ["Annotation Free", "Scarce Annotation", "Domain Supervisor"]

glove_results = {}
bert_results = {}
for k in range(3):
    seeds_glove = None
    for subdir, dirs, files in os.walk(glove_dir+setting_dirs[k]):
        for file in files:
            if file == 'bootstraps.pt':
                metrics = torch.load(os.path.join(subdir, file))
                if seeds_glove is None:
                    seeds_glove = metrics
                else:
                    for i, (key, value) in enumerate(seeds_glove.items()):
                        seeds_glove[key] = np.concatenate([value, metrics[key]])
    glove_results[settings_name_map[k]] = seeds_glove
    seeds_bert = None
    for subdir, dirs, files in os.walk(bert_dir + setting_dirs[k]):
        for file in files:
            if file == 'bootstraps.pt':
                metrics = torch.load(os.path.join(subdir, file))
                if seeds_bert is None:
                    seeds_bert = metrics
                else:
                    for i, (key, value) in enumerate(seeds_bert.items()):
                        seeds_bert[key] = np.concatenate([value, metrics[key]])
    bert_results[settings_name_map[k]] = seeds_bert


results_dir = "/Users/mob/Desktop/plots/"
df = pd.DataFrame(columns=['ACC', 'HA', 'ebF1', 'miF1', 'maF1', 'SUPERVISION', 'GRAPH'])
for key in ['ACC', 'HA', 'ebF1', 'miF1', 'maF1']:
    for k in settings_name_map:
        g = pd.DataFrame(glove_results[k])
        g['SUPERVISION'] = [k for i in range(len(g))]
        g['GRAPH'] = ["GloVe" for i in range(len(g))]
        df = df.append(g)
        b = pd.DataFrame(bert_results[k])
        b['SUPERVISION'] = [k for i in range(len(b))]
        b['GRAPH'] = ["Bert" for i in range(len(b))]
        df = df.append(b)

for key in ['ACC', 'HA', 'ebF1', 'miF1', 'maF1']:
    plt.plot()
    sns.set_theme()
    sns.boxplot(data=df, x="SUPERVISION", y=key, hue="GRAPH")
    plt.tight_layout()
    plt.savefig(results_dir + key + '.pdf')
    plt.close()
    # plt.show()

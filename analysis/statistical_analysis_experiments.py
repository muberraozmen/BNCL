import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats


df = pd.read_csv('reuters_cluster_results.csv')
sns.set_theme(palette="Set2", style="white", font_scale=2)


def plot_metric(metric_name):
    plt.plot()
    sns.violinplot(data=df, x="K", y=metric_name, split=True)
    plt.xlabel("Number of Label Groups")
    plt.ylabel("")
    plt.title(metric_name)
    plt.tight_layout()
    plt.savefig(metric_name+'.pdf')
    plt.show()


for metric in ["HA", "ACC", "ebF1", "miF1", "maF1"]:
    plot_metric(metric)


print(df.groupby("K").mean().transpose())
print(df.groupby("K").std().transpose())

confidence = 0.95
n = 50
coeff = scipy.stats.t.ppf((1 + confidence) / 2., n-1)




from sklearn.cluster import DBSCAN
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("https://reneshbedre.github.io/assets/posts/tsne/tsne_scores.csv")

clusters = DBSCAN(eps=2.5, min_samples=4).fit(df)

p = sns.scatterplot(data=df, x="t-SNE-1", y="t-SNE-2", hue=clusters.labels_, legend="full", palette="deep")
sns.move_legend(p, "upper right", bbox_to_anchor=(1.17, 1.2), title='Clusters')
plt.savefig("sklearn_dbscan.png")
plt.show()
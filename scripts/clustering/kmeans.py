from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.cluster import silhouette_visualizer


class Kmeans:
    matrix = None

    def __init__(self, matrix):
        self.matrix = matrix

    def find_k(self, max_k):
        iters = range(2, max_k+1)
        
        sse = []
        silhouette = []
        
        for k in iters:
            model = self.model(n_clusters=k)
            sse.append(model.fit(self.matrix).inertia_)
            silhouette.append(silhouette_score(self.matrix, model.labels_))
            print('Fit {} clusters'.format(k))
            
        f, ax = plt.subplots(1, 1)
        ax.plot(iters, sse, marker='o')
        ax.set_xlabel('Cluster Centers')
        ax.set_xticks(iters)
        ax.set_xticklabels(iters)
        ax.set_ylabel('SSE')
        ax.set_title('SSE by Cluster Center Plot')
        plt.tight_layout()
        plt.show()

        f, ax = plt.subplots(1, 1)
        ax.plot(iters, silhouette, marker='o')
        ax.set_xlabel('Cluster Centers')
        ax.set_xticks(iters)
        ax.set_xticklabels(iters)
        ax.set_ylabel('Score')
        ax.set_title('Silhoute Score by Cluster')
        plt.tight_layout()
        plt.show()

    def create_hierarchical_clustering(self):
        sns.set_style("whitegrid")
        clustermap = sns.clustermap(self.matrix, figsize=(100,80))
        plt.tight_layout()
        # plt.show()
        # clustermap.savefig("out.png")

    def find_silhouette(self, k):
        silhouette_visualizer(KMeans(k), self.matrix, colors='yellowbrick')
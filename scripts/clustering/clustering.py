from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.cluster import silhouette_visualizer



def find_k(data, max_k):
    iters = range(2, max_k+1)
    
    sse = []
    silhouette = []
    for k in iters:
        km = KMeans(n_clusters=k)
        sse.append(km.fit(data).inertia_)
        silhouette.append(silhouette_score(data, km.labels_))
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


def create_hierarchical_clustering(data):
    sns.set_style("whitegrid")
    clustermap = sns.clustermap(data,figsize=(100,80))
    plt.tight_layout()
    # plt.show()
    # clustermap.savefig("out.png") 

def find_silhouette(data, k):
    silhouette_visualizer(KMeans(k), data, colors='yellowbrick')
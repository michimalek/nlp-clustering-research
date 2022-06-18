from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.cluster import silhouette_visualizer, KElbowVisualizer

# Script to pass sklearn model (K-Means, Agglomerative Clustering, Spectral Clustering)
# and embedding to generate statistics for finding optimal number of clusters for given algorithm
class KFinder:
    model = None
    matrix = None

    def __init__(self, model, matrix):
        self.model = model
        self.matrix = matrix

    def find_k(self, max_k):
        """Loops through iters range of 2 and max_k+1 and returns statstics of silhouete, calinski and davies scores
        for each iteration"""
        iters = range(2, max_k+1)
        sse = []
        silhouette = []
        calinski = []
        davies = []
        
        for k in iters:
            temp_model = None
            if isinstance(self.model, KMeans):
                temp_model = KMeans(k, random_state=420).fit(self.matrix)
                sse.append(temp_model.inertia_)

            if isinstance(self.model, AgglomerativeClustering):
                temp_model = AgglomerativeClustering(k).fit(self.matrix)

            if isinstance(self.model, SpectralClustering):
                temp_model = SpectralClustering(k).fit(self.matrix)

            if temp_model is None:
                print("The given model was not recognized. Available models: KMeans, Agglomerative Clustering, Spectral Clustering")
                return
                
            silhouette.append(silhouette_score(self.matrix, temp_model.labels_))
            calinski.append(calinski_harabasz_score(self.matrix, temp_model.labels_))
            davies.append(davies_bouldin_score(self.matrix, temp_model.labels_))

            print('Fit {} clusters'.format(k))
            
        if isinstance(self.model, KMeans):
            visualizer = KElbowVisualizer(self.model, k=(1,max_k))
            visualizer.fit(self.matrix)   
            visualizer.show()  

        f, ax = plt.subplots(3)
        ax[0].plot(iters, silhouette, marker='o')
        ax[0].set_xlabel('Cluster Centers')
        ax[0].set_xticks(iters)
        ax[0].set_xticklabels(iters)
        ax[0].set_ylabel('Score')
        ax[0].set_title('Silhoute Score by Cluster')

        ax[1].plot(iters, calinski, marker='o')
        ax[1].set_xlabel('Cluster Centers')
        ax[1].set_xticks(iters)
        ax[1].set_xticklabels(iters)
        ax[1].set_ylabel('Score')
        ax[1].set_title('Calinski Harabasz Score by Cluster')

        ax[2].plot(iters, davies, marker='o')
        ax[2].set_xlabel('Cluster Centers')
        ax[2].set_xticks(iters)
        ax[2].set_xticklabels(iters)
        ax[2].set_ylabel('Score')
        ax[2].set_title('Davies Bouldin Score by Cluster')

        plt.tight_layout()
        plt.show()
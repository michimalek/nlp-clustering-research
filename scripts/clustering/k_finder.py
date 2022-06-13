from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.cluster import silhouette_visualizer, KElbowVisualizer

class KFinder:
    model = None
    matrix = None

    def __init__(self, model, matrix):
        self.model = model
        self.matrix = matrix

    def find_k(self, max_k):
        iters = range(2, max_k+1)
        sse = []
        silhouette = []
        
        for k in iters:
            temp_model = None
            if isinstance(self.model, KMeans):
                temp_model = KMeans(k).fit(self.matrix)
                sse.append(temp_model.inertia_)

            if isinstance(self.model, AgglomerativeClustering):
                temp_model = AgglomerativeClustering(k).fit(self.matrix)

            if isinstance(self.model, SpectralClustering):
                temp_model = SpectralClustering(k).fit(self.matrix)

            if temp_model is None:
                print("The given model was not recognized. Available models: KMeans, Agglomerative Clustering, Spectral Clustering")
                return
                
            silhouette.append(silhouette_score(self.matrix, temp_model.labels_))
            print('Fit {} clusters'.format(k))
            
        if isinstance(self.model, KMeans):
            # f, ax = plt.subplots(1, 1)
            # ax.plot(iters, sse, marker='o')
            # ax.set_xlabel('Cluster Centers')
            # ax.set_xticks(iters)
            # ax.set_xticklabels(iters)
            # ax.set_ylabel('SSE')
            # ax.set_title('SSE by Cluster Center Plot')
            # plt.tight_layout()
            # plt.show()
            visualizer = KElbowVisualizer(self.model, k=(1,max_k))
            visualizer.fit(self.matrix)   
            visualizer.show()  

        f, ax = plt.subplots(1, 1)
        ax.plot(iters, silhouette, marker='o')
        ax.set_xlabel('Cluster Centers')
        ax.set_xticks(iters)
        ax.set_xticklabels(iters)
        ax.set_ylabel('Score')
        ax.set_title('Silhoute Score by Cluster')
        plt.tight_layout()
        plt.show()

    def find_silhouette(self, k):
        silhouette_visualizer(KMeans(k), self.matrix, colors='yellowbrick')
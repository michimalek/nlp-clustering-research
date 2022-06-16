from sklearn.metrics import silhouette_score, calinski_harabasz_score
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
        calinski = []
        
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

            print('Fit {} clusters'.format(k))
            
        if isinstance(self.model, KMeans):
            visualizer = KElbowVisualizer(self.model, k=(1,max_k))
            visualizer.fit(self.matrix)   
            visualizer.show()  

        f, ax = plt.subplots(2)
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

        plt.tight_layout()
        plt.show()

    def find_silhouette(self, k):
        silhouette_visualizer(KMeans(k), self.matrix, colors='yellowbrick')
from bertopic import BERTopic
import numpy
from hdbscan import HDBSCAN
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, AffinityPropagation


class TopicVis:
    model = None
    topics = None
    probs = None
    sentences = None
    cluster_method = None
    

    def __init__(self, sentences, cluster_method=None, k=None):
        self.cluster_method = cluster_method
        if cluster_method is None:
            cluster_method = HDBSCAN(metric='euclidean', cluster_selection_method='eom', prediction_data=True)
            print("Standard Clustering HDBSCAN selected")
        
        if k is not None:
            if isinstance(cluster_method, KMeans):
                self.cluster_method = KMeans(k)
                print("Alternative Clustering with set k successfully selected")

            if isinstance(cluster_method, AgglomerativeClustering):
                self.cluster_method = AgglomerativeClustering(k)
                print("Alternative Clustering with set k successfully selected")

            if isinstance(cluster_method, SpectralClustering):
                self.cluster_method = SpectralClustering(k)
                print("Alternative Spectral Clustering with set k successfully selected")
            
            self.model = BERTopic(embedding_model="sentence-t5-xl", hdbscan_model=cluster_method, nr_topics=k)

        
        else: 
            self.cluster_method = cluster_method
            self.model = BERTopic(embedding_model="sentence-t5-xl", hdbscan_model=cluster_method, nr_topics="auto")
            print("Alternative Clustering without set k successfully selected")

        
        print("finishing process..")
        self.sentences = sentences
        self.opics, self.probs = self.model.fit_transform(self.sentences)

    def get_topics_probs(self):
        return (self.topics, self.probs)

    def show_topics(self):
        fig = self.model.visualize_topics()
        fig.show()

    def get_info(self):
        print(self.model.get_topic_info())

    def show_bars(self):
        fig = self.model.visualize_barchart()
        fig.show()

    def show_heatmap(self):
        fig = self.model.visualize_heatmap()
        fig.show()

    def show_hierarchy(self):
        fig = self.model.visualize_hierarchy()
        fig.show()

    def get_embedding(self):
        self.model.encode
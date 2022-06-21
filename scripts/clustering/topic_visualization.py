from bertopic import BERTopic
import numpy
from hdbscan import HDBSCAN
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from metrics.unsupervised import silhouette_score, get_calinski_score


class TopicVis:
    model = None
    topics = None
    probs = None
    sentences = None
    cluster_method = None
    

    def __init__(self, sentences, cluster_method=None, k=None):
        self.cluster_method = cluster_method
        if cluster_method is None:
            cluster_method = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        
        if isinstance(k, int):
            if isinstance(self.model, KMeans):
                self.cluster_method = KMeans(k, random_state=420).fit(self.matrix)

            if isinstance(self.model, AgglomerativeClustering):
                self.cluster_method = AgglomerativeClustering(k).fit(self.matrix)

            if isinstance(self.model, SpectralClustering):
                self.cluster_method = SpectralClustering(k).fit(self.matrix)
            
            self.model = BERTopic(embedding_model="sentence-t5-xl", hdbscan_model=cluster_method, nr_topics=k)

        
        else: 
            self.cluster_method = cluster_method
            self.model = BERTopic(embedding_model="sentence-t5-xl", hdbscan_model=cluster_method, nr_topics="auto")

            print("Alternative Clustering successfully selected")

        self.sentences = sentences
        self.opics, self.probs = self.model.fit_transform(self.sentences)

        print(self.model.embedding_model)
    
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
from bertopic import BERTopic
import numpy
from flair.embeddings import TransformerDocumentEmbeddings
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from metrics.unsupervised import silhouette_score, get_calinski_score


class TopicVis:
    model = None
    topics = None
    probs = None
    sentences = None
    cluster_method = None
    

    def __init__(self, sentences, cluster_method=None):
        # model_name = TransformerDocumentEmbeddings('distilbert-base-uncased-finetuned-sst-2-english')
        self.cluster_method = cluster_method
        if cluster_method is None:
            cluster_method = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        else: 
            self.cluster_method = cluster_method
            print("Alternative Clustering successfully selected")
        self.model = BERTopic(embedding_model="all-mpnet-base-v2", hdbscan_model=cluster_method, nr_topics="auto")
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
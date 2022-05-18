from bertopic import BERTopic
import numpy

class cluster_visualization:
    model = None
    topics = None
    probs = None
    sentences = None

    def __init__(self, sentences):
        self.model = BERTopic()
        self.sentences = sentences
        self.opics, self.probs = self.model.fit_transform()
    
    def get_topics_probs(self):
        return (self.topics, self.probs)

    def show_topics(self):
        print(self.model.get_topic_info())
        fig = self.model.visualize_topics()
        fig.show()

    def get_info(self):
        print(self.model.get_topic_info())

    def show_bars(self):
        fig = self.model.visualize_barchart()
        fig.show()
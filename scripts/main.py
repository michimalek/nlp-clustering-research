from fastcore.transform import Pipeline
# from pipeline_functions import remove_spaces, remove_special_chars, lowercase
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from embeddings.tfidf import get_df, get_matrix
from embeddings.preprocessing import to_set, prepare_sentences
from clustering.topic_visualization import TopicVis
from clustering.kmeans import Kmeans
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, AffinityPropagation

def kmeans_find_k(clean_sentences, max_k):
    tfidf_matrix = get_matrix(clean_sentences)
    Kmeans(tfidf_matrix).find_k(max_k)

def tfidf_kmeans_pipeline(clean_sentences):
    return get_matrix(clean_sentences)

def bert_hdbscan_pipeline(clean_words):
    TopicVis(clean_words).show_bars()

def bert_kmeans_pipeline(clean_sentences, n_clusters):
    TopicVis(clean_sentences, KMeans(n_clusters=n_clusters)).show_bars()

def bert_agglomerative_pipeline(clean_sentences, n_clusters):
    TopicVis(clean_sentences,AgglomerativeClustering(n_clusters=n_clusters)).show_bars()

def bert_meanshift_pipeline(clean_sentences):
    TopicVis(clean_sentences, MeanShift()).show_bars()

def bert_affinity_pipeline(clean_sentences):
    TopicVis(clean_sentences, AffinityPropagation()).show_bars()

if __name__ == '__main__':
    data = to_set(pd.read_csv("data/Recommendations.csv")["Recommendation"])
    clean = prepare_sentences(data)
    bert_hdbscan_pipeline(clean)
    
    # model = Kmeans(get_df(to_set(prepare_sentences(data))).toarray())
    # model.find_k(10)
    # visual = topic_visualization(to_set(prepare_sentences(data)))
    # visual.show_topics()
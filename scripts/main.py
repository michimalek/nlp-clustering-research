from fastcore.transform import Pipeline
# from pipeline_functions import remove_spaces, remove_special_chars, lowercase
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from embeddings.tfidf import get_df, get_matrix
from embeddings.preprocessing import to_set, prepare_sentences
from clustering.topic_visualization import TopicVis
from clustering.kmeans import Kmeans
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, AffinityPropagation
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

def kmeans_find_k(clean_sentences, max_k):
    tfidf_matrix = get_matrix(clean_sentences)
    Kmeans(tfidf_matrix).find_k(max_k)

def tfidf_kmeans_pipeline(clean_sentences):
    return get_matrix(clean_sentences)

def transformer_kmeans_pipeline(clean_sentences):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(clean_sentences)
    umap = UMAP()
    umap_embeddings = umap.fit_transform(embeddings)
    Kmeans(umap_embeddings).find_k(70)

def transformer_hdbscan_pipeline(clean_sentences):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(clean_sentences)
    umap = UMAP()
    umap_embeddings = umap.fit_transform(embeddings)
    hdb = HDBSCAN()
    hdb.fit(umap_embeddings)
    return hdb.labels_

def transformer_meanshift_pipeline(clean_sentences):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(clean_sentences)
    umap = UMAP()
    umap_embeddings = umap.fit_transform(embeddings)
    mshift = MeanShift()
    mshift.fit(umap_embeddings)
    return mshift.labels_

def transformer_affinity_pipeline(clean_sentences):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(clean_sentences)
    umap = UMAP()
    umap_embeddings = umap.fit_transform(embeddings)
    affinity = AffinityPropagation()
    affinity.fit(umap_embeddings)
    return affinity.labels_

def bert_hdbscan_pipeline(clean_sentences):
    TopicVis(clean_sentences).show_bars()

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
    # bert_meanshift_pipeline(clean)
    bert_affinity_pipeline(clean)

from fastcore.transform import Pipeline
# from pipeline_functions import remove_spaces, remove_special_chars, lowercase
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from embeddings.tfidf import get_df, get_matrix
from embeddings.preprocessing import remove_duplicates, prepare_sentences
from clustering.topic_visualization import TopicVis
from clustering.kmeans import Kmeans
from clustering.k_finder import KFinder
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, AffinityPropagation, SpectralClustering
from sklearn.metrics import silhouette_score
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

def kmeans_find_k(clean_sentences, max_k):
    tfidf_matrix = get_matrix(clean_sentences)
    Kmeans(tfidf_matrix).find_k(max_k)

def tfidf_kmeans_pipeline(clean_sentences):
    return get_matrix(clean_sentences)

def transformer_kmeans_pipeline_find_k(clean_sentences, max_k):
    model = SentenceTransformer('sentence-t5-xl')
    embeddings = model.encode(clean_sentences)
    umap = UMAP()
    umap_embeddings = umap.fit_transform(embeddings)
    KFinder(KMeans(), umap_embeddings).find_k(max_k)

def transformer_kmeans_pipeline(clean_sentences, n_clusters):
    model = SentenceTransformer('sentence-t5-xl')
    embeddings = model.encode(clean_sentences)
    umap = UMAP()
    umap_embeddings = umap.fit_transform(embeddings)
    kmeans = KMeans(n_clusters= n_clusters).fit(umap_embeddings)
    print(silhouette_score(umap_embeddings, kmeans.labels_))
    return kmeans.labels_

def transformer_agglomerative_pipeline_find_k(clean_sentences, max_k):
    model = SentenceTransformer('sentence-t5-xl')
    embeddings = model.encode(clean_sentences)
    umap = UMAP()
    umap_embeddings = umap.fit_transform(embeddings)
    KFinder(AgglomerativeClustering(), umap_embeddings).find_k(max_k)

def transformer_agglomerative_pipeline(clean_sentences, n_clusters):
    model = SentenceTransformer('sentence-t5-xl')
    embeddings = model.encode(clean_sentences)
    umap = UMAP()
    umap_embeddings = umap.fit_transform(embeddings)
    agglomerative = AgglomerativeClustering(n_clusters= n_clusters).fit(umap_embeddings)
    print(silhouette_score(umap_embeddings, agglomerative.labels_))
    return agglomerative.labels_

def transformer_spectral_pipeline_find_k(clean_sentences, max_k):
    model = SentenceTransformer('sentence-t5-xl')
    embeddings = model.encode(clean_sentences)
    umap = UMAP()
    umap_embeddings = umap.fit_transform(embeddings)
    KFinder(SpectralClustering(), umap_embeddings).find_k(max_k)

def transformer_spectral_pipeline(clean_sentences, n_clusters):
    model = SentenceTransformer('sentence-t5-xl')
    embeddings = model.encode(clean_sentences)
    umap = UMAP()
    umap_embeddings = umap.fit_transform(embeddings)
    spectral = SpectralClustering(n_clusters= n_clusters).fit(umap_embeddings)
    print(silhouette_score(umap_embeddings, spectral.labels_))
    return spectral.labels_

def transformer_hdbscan_pipeline(clean_sentences):
    model = SentenceTransformer('sentence-t5-xl')
    embeddings = model.encode(clean_sentences)
    umap = UMAP()
    umap_embeddings = umap.fit_transform(embeddings)
    hdb = HDBSCAN().fit(umap_embeddings)
    return hdb.labels_

def transformer_meanshift_pipeline(clean_sentences):
    model = SentenceTransformer('sentence-t5-xl')
    embeddings = model.encode(clean_sentences)
    umap = UMAP()
    umap_embeddings = umap.fit_transform(embeddings)
    mshift = MeanShift()
    mshift.fit(umap_embeddings)
    return mshift.labels_

def transformer_affinity_pipeline(clean_sentences):
    model = SentenceTransformer('sentence-t5-xl')
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
    TopicVis(clean_sentences, AgglomerativeClustering(n_clusters=n_clusters)).show_bars()

def bert_meanshift_pipeline(clean_sentences):
    TopicVis(clean_sentences, MeanShift()).show_bars()

def bert_affinity_pipeline(clean_sentences):
    TopicVis(clean_sentences, AffinityPropagation()).show_bars()

if __name__ == '__main__':
    data = remove_duplicates(pd.read_excel("data/Recommendations_label.xlsx")["Recommendation"])
    clean = prepare_sentences(data)
    labels = transformer_kmeans_pipeline_find_k(clean, 20)
    # df = pd.DataFrame({"Recommendation": data, "Label": labels})
    # df.to_excel("results_new/kmeans_7_result.xlsx")
    # print(pd)
    # print(np.unique(labels))

    


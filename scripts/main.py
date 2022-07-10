import pandas as pd
import numpy as np
import os
from embeddings.tfidf import get_df, create_matrix
from embeddings.sbert import create_embedding
from embeddings.preprocessing import remove_duplicates, prepare_sentences, return_feature_sentence_length
from clustering.topic_visualization import TopicVis
from clustering.k_finder import KFinder
from clustering.external_validation import get_validation_df, get_validation_txt
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, AffinityPropagation, SpectralClustering
from sklearn.metrics import silhouette_score
from umap import UMAP
from hdbscan import HDBSCAN

def transformer_kmeans_pipeline_find_k(clean_sentences, max_k):
    # sentence_length = return_feature_sentence_length(clean_sentences)
    embedding = create_embedding(clean_sentences)
    # Adds additional features to embedding, in this case the feature sentence_length
    # embedding = np.c_[embedding, sentence_length]
    umap = UMAP()
    umap_embedding = umap.fit_transform(embedding)
    KFinder(KMeans(), umap_embedding).find_k(max_k)

def transformer_kmeans_pipeline(clean_sentences, n_clusters):
    embedding = create_embedding(clean_sentences)
    umap = UMAP()
    umap_embedding = umap.fit_transform(embedding)
    kmeans = KMeans(n_clusters= n_clusters, random_state=25).fit(umap_embedding)
    return kmeans.labels_

def transformer_agglomerative_pipeline_find_k(clean_sentences, max_k):
    embedding = create_embedding(clean_sentences)
    umap = UMAP()
    umap_embedding = umap.fit_transform(embedding)
    KFinder(AgglomerativeClustering(), umap_embedding).find_k(max_k)

def transformer_agglomerative_pipeline(clean_sentences, n_clusters):
    embedding = create_embedding(clean_sentences)
    umap_embedding = UMAP().fit_transform(embedding)
    agglomerative = AgglomerativeClustering(n_clusters= n_clusters).fit(umap_embedding)
    print(silhouette_score(umap_embedding, agglomerative.labels_))
    return agglomerative.labels_

def transformer_spectral_pipeline_find_k(clean_sentences, max_k):
    embedding = create_embedding(clean_sentences)
    umap = UMAP()
    umap_embedding = umap.fit_transform(embedding)
    KFinder(SpectralClustering(), umap_embedding).find_k(max_k)

def transformer_spectral_pipeline(clean_sentences, n_clusters):
    embedding = create_embedding(clean_sentences)
    umap = UMAP()
    umap_embedding = umap.fit_transform(embedding)
    spectral = SpectralClustering(n_clusters= n_clusters).fit(umap_embedding)
    print(silhouette_score(umap_embedding, spectral.labels_))
    return spectral.labels_

def transformer_hdbscan_pipeline(clean_sentences):
    embedding = create_embedding(clean_sentences)
    umap = UMAP(random_state=25)
    umap_embedding = umap.fit_transform(embedding)
    hdb = HDBSCAN().fit(umap_embedding)
    return hdb.labels_

def transformer_meanshift_pipeline(clean_sentences):
    embedding = create_embedding(clean_sentences)
    umap = UMAP()
    umap_embedding = umap.fit_transform(embedding)
    mshift = MeanShift()
    mshift.fit(umap_embedding)
    return mshift.labels_

def transformer_affinity_pipeline(clean_sentences):
    embedding = create_embedding(clean_sentences)
    umap = UMAP()
    umap_embedding = umap.fit_transform(embedding)
    affinity = AffinityPropagation(random_state=25)
    affinity.fit(umap_embedding)
    return affinity.labels_

def create_external_validation_txt(true, predicted, dir="results"):
    open('metrices.txt', 'w').close()

    for subdir, dirs, files in os.walk(dir):
        for file in files:
            filepath = subdir + os.sep + file        
            pred = pd.read_excel(filepath)["Label"]    
            with open('external_validation.txt', 'a') as f:
                f.write('------------------------------\n')
                f.write(f'Name: {file}\n\n')
                f.writelines(get_validation_txt(true, pred))
                f.write('\n')

def create_external_validation_df(true, predicted, dir = "results"):
    df = pd.DataFrame()
    i = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            filepath = subdir + os.sep + file
            df = pd.concat([df, get_validation_df(true,predicted)],ignore_index=True)
            i.append(file.split(".")[0])
            print(i)
    df.columns = ["Rand Index","Homogeneity Score","Completness Score","V-Measure","Purity"]
    df["algo_name"] = i
    df.set_index("algo_name", inplace=True)
    return df

def create_excel(path, sentences, labels):
    df = pd.DataFrame({"Recommendation": sentences, "Label": labels})
    df.to_excel(path + ".xlsx")

if __name__ == '__main__':
    # Read data
    data = pd.read_excel("data/Recommendations_label.xlsx")["Recommendation"]

    # Clean data
    clean = prepare_sentences(data)

    # Produce labels with given clustering algorithm
    labels = transformer_spectral_pipeline(clean, 7)

    # Save Excel with original sentences and predicted corresponding labels to given path
    # create_excel("results/spectral_7", data, labels)
    


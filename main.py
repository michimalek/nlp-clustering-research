from fastcore.transform import Pipeline
# from pipeline_functions import remove_spaces, remove_special_chars, lowercase

import pandas as pd
import numpy

import matplotlib.pyplot as plt
import seaborn as sns
from scripts.tfidf import get_df, get_matrix, to_set, prepare_sentences
from scripts.word2vec import load_sentence_transformer
from scripts import ClusterVisualization
from scripts.clustering import find_k, create_hierarchical_clustering, find_silhouette




def kmeans_pipeline(data):
    # Creates a pipeline with a list of functions
    pipe = Pipeline([get_df, find_k])
    
    # Invokes pipeline
    output = pipe(data)

    return output


if __name__ == '__main__':
    data = pd.read_csv("data/Recommendations.csv")
    # vectors = load_sentence_transformer(to_set(prepare_sentences(data)))
    # find_silhouette(vectors, 39)
    # find_k(vectors, 100)
    visual = ClusterVisualization(to_set(prepare_sentences(data)))
    visual.show_topics()

    # max_k = 20
    # for k in range(max_k):
    #     if k > 1:
    #         find_silhouette(matrix_array, k)
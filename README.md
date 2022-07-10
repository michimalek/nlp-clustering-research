# NLP-clustering for Recommendation

## Introduction
This library focuses on sentence clustering, specifically recommendation clustering for telecommunication networks, but can be applied to any sort of sentences. It is based on the bachelor research "Textual Clusterinf for Telecommunication Accident Recommendations". The recommendations used in this specific case were extracted during workshops for accident analysis in the research of [Wienen et al.]()

## How it works
The following methodology is applied:
1. Clean sentences from punctuation and stop words
2. Quantify textual sentences into an numerical multi-dimensional matrix (embedding), each row representing the original sentence and and each column representing the feature values set by the embedding method. [S-Bert](https://www.sbert.net/index.html) was used to create the embedding.
3. As beforementioned, the produced embedding matrix includes a tremendous amount of dimensions. To handle the processing faster and simplify the data, the dimensionality reduction tools UMAP will be utilized to transform the multi-dimensional matrix into two-dimnesional space. [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) was used to transform the matrix into two-dimensional space.
4. After simpifing the matrix, the clustering can be conducted with a variety of machine learning methods. The following clustering methods are available for this library:
    - K-Means
    - Agglomerative Clustering
    - Spectral Clustering
    - Affinity Propagation
    - Mean Shift
    - HDBSCAN

All the aforementioned steps 1. - 4. are processed in the *main.py* file. There are more jupyter notebooks for label simularity (*accuary.ipynb*), internal validation methods to find the optimal number of clusters for a specfic model (*internal_validation.ipynb*) and cluster visulaization and word cloud generation for each cluster (*internal_validation.ipynb*).

## Usage
To receive the labeled data from the clustering models, use *main.py*. It includes all the important data-pipelines to get from only text to text including predicted labels.

### Example with K-Means algorithm
To conduct the labels generated based on K-Means, replace the *algorithm* at `transformer_<algorithm>_pipeline` in *main.py* with one of the aglorithms mentioned above. The exact name of the functions can be found in the *main.py* as well. 
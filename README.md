# NLP-clustering for Recommendation

## Introduction
This library focuses on sentence clustering, specifically recommendation clustering for telecommunication networks, but can be applied to any sort of sentences. It is based on the bachelor research "Textual Clustering for Telecommunication Accident Recommendations". The recommendations used in this specific case were extracted during workshops for accident analysis in the research of [Wienen et al.]() This library focuses on finding the most optimal unsupervised text clustering method with the best number of clusters. This consists firstly on internal validation scores to find the most optimal number of cluster and secondly on external validation indices to find the most optimal out of all trained models. It also allows the visual comparison of different algortihms based on dimensionality reduction.

## How it works
The following methodology is applied:
1. Clean sentences from punctuation and stop words
2. Quantify textual sentences into an numerical multi-dimensional matrix (embedding), each row representing the original sentence and each column representing the feature values set by the embedding method. [S-Bert](https://www.sbert.net/index.html) was used to create the embedding.
3. As beforementioned, the produced embedding matrix includes a tremendous amount of dimensions. To handle the processing faster and simplify the data, the dimensionality reduction tool [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) is utilized to transform the multi-dimensional matrix into two-dimnesional space.
4. After simpifing the matrix, the clustering can be conducted with a variety of machine learning methods. The following clustering methods are available for this library:
    - Manual-K:
        - K-Means
        - Agglomerative Clustering
        - Spectral Clustering
    - Auto-K:
        - Affinity Propagation 
        - Mean Shift
        - HDBSCAN

Manual-K algorithms distinguish themselves from auto-K algorithms, because manual-k requires a given optimal number of clusters as paramemter and auto-k methods not. Instead, they calculate the optimal number of clusters based on their own internal calculations. Therefore, to find the optimal number of clusters K for the manual-K methods, the KFinder objects gets used. It iterates through a given range and calculates internal validation indices for each iteration to indicate which K performed the best.

All the aforementioned steps 1. - 4. are processed in the *main.py* file. There are more jupyter notebooks for label simularity (*accuary.ipynb*), internal validation methods to find the optimal number of clusters for a specfic model (*internal_validation.ipynb*) and cluster visulaization and word cloud generation for each cluster (*internal_validation.ipynb*).

## Usage
To receive the labeled data from the clustering models, use *main.py*. It includes all the important data-pipelines to get from only text to text including predicted labels.

### Example with K-Means algorithm
To conduct the labels generated based on K-Means, first you have to find the optimal number of clusters K for K-Means. This can be done with the KFinder object, either in the *main.py* or the *internal_validation.ipynb*; the format of the function would be `transformer_kmeans_pipeline_find_k(clean)`, with the only parameter being the clean sentences. After running this method, it returns the internal indices scores over the number of given iterations, the person using this script then has to evaluate K by himself. 

After finding the optimal K (potentially also multiple K's with similar internal validation scores), get the predicted labels of the trained model of K-Means with `labels = transformer_kmeans_pipeline(clean, K)` in *main.py* and safe the excel with the original text and the corresponding labels. The parameters depend on the algorithm used; Manual-K method obviously require the cleaned data and the number of K, auto-K methods only inlcude the cleaned data.

After the K-Mean model is trained and produced the clusters, they are probably sufficiently working but there are of course more algorithms available for clustering. In the case switching to a different algorithm, the aforementioned methods also be used with different algorithms like `transformer_spectral_pipeline_find_k(clean)` or `transformer_agglomerative_pipeline_find_k(clean)` for finding the optimal K and `transformer_spectral_pipeline(clean)` or `transformer_agglomerative_pipeline(clean)` for producing the actual clusters. Check out the notebooks for validation and visualization methods.

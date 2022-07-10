# NLP-clustering for Recommendation

## Introduction
This library focuses on sentence clustering, specifically recommendation clustering for telecommunication networks, but can be applied to any sort of sentences. It is based on the bachelor research "Textual Clusterinf for Telecommunication Accident Recommendations".

## How does it work
The following methodology is applied:
1. Clean sentences from punctuation and stop words
2. Quantify textual sentences into an numerical multi-dimensional matrix (embedding), each row representing the original sentence and and column representing the feature values set by the embedding method. [S-Bert](https://www.sbert.net/index.html) was used to create the embedding.
3. As beforementioned, the produced embedding matrix includes a tremendous amount of dimensions. To handle the processing faster and simplify the data, the dimensionality reduction tools UMAP will be utilized to transform the multi-dimensional matrix into two-dimnesional space.
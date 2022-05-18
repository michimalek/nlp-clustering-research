from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def load_sentence_transformer(sentences):
    """
    This is a simple application for sentence embeddings: clustering
    Sentences are mapped to sentence embeddings and then k-mean clustering is applied.
    """
    

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Corpus with example sentences
    corpus_embeddings = embedder.encode(sentences)

    return corpus_embeddings
    # clustered_sentences = [[] for i in range(num_clusters)]
    # for sentence_id, cluster_id in enumerate(cluster_assignment):
    #     clustered_sentences[cluster_id].append(sentences[sentence_id])

    # for i, cluster in enumerate(clustered_sentences):
    #     print("Cluster ", i+1)
    #     print(cluster)
    #     print("")

    return
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # sentence_embeddings = model.encode(sentences)

    # for sentence, embedding in zip(sentences, sentence_embeddings):
    #     print("Sentence:", sentence)
    #     print("Embedding:", embedding)
    #     print("")

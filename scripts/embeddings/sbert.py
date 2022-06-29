from sentence_transformers import SentenceTransformer
from umap import UMAP

def create_embedding(clean_sentences, pretrained_model = "sentence-t5-xl"):
    model = SentenceTransformer(pretrained_model)
    return model.encode(clean_sentences)

def create_umap_embedding(clean_sentences, pretrained_model = "sentence-t5-xl"):
    model = SentenceTransformer(pretrained_model)
    return UMAP().fit_transform(model.encode(clean_sentences))
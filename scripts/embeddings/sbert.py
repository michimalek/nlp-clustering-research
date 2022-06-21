from sentence_transformers import SentenceTransformer

def create_embedding(clean_sentences, pretrained_model = "sentence-t5-xl"):
    model = SentenceTransformer(pretrained_model)
    return model.encode(clean_sentences)
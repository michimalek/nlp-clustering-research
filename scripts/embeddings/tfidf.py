import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def create_matrix(data):
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(data)

def get_df(data):
    return pd.DataFrame(create_matrix(data).toarray())

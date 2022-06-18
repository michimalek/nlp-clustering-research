import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def get_matrix(data):
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(data)

def get_df(data):
    return pd.DataFrame(get_matrix(data).toarray())

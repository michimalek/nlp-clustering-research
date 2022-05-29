import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')

def get_matrix(data):
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(data)

def get_df(data):
    return pd.DataFrame(get_matrix(data).toarray())

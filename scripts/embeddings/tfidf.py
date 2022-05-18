import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')

def process_text(text) -> list:
    # Takes in a string of text, then performs the following:
    # 1. Remove all punctuation
    # 2. Remove all stopwords
    # 3. Return the cleaned text as a list of words
    # 4. Remove words
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc =  [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
    return [stemmer.lemmatize(word) for word in nopunc]

def to_set(data):
    return list(set(data))

def prepare_sentences(data) -> list:
    docs = []
    for sentence in data.iloc[:, 0]:
        docs.append(" ".join(process_text(sentence)))
    return docs

def prepare_words(data) -> list:
    words = []
    for rec in data.iloc[:, 0]:
        words.append(process_text(rec))
    return words

def get_matrix(data):
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(to_set(prepare_sentences(data)))

def get_df(data):
    return pd.DataFrame(get_matrix(data).toarray())

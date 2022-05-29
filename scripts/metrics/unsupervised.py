from sklearn.metrics import silhouette_score, calinski_harabasz_score

def get_silhouete_score(data, labels):
    return silhouette_score(data, labels, metric='euclidean')

def get_calinski_score(data, labels):
    return calinski_harabasz_score(data, labels)
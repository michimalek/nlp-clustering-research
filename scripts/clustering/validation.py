import numpy as np
import pandas as pd
from sklearn.metrics import rand_score, homogeneity_score, completeness_score, v_measure_score, accuracy_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn import metrics

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # calculate and return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def get_validation_txt(true_labels, pred_labels):
    return [f"Rand Index (0-1): {rand_score(true_labels, pred_labels)}\n",
        f"Adj. Rand Index: {adjusted_rand_score(true_labels, pred_labels)}\n",
        f"Homogeneity Score: {homogeneity_score(true_labels, pred_labels)}\n",
        f"Completness Score: {completeness_score(true_labels, pred_labels)}\n",
        f"V-Measure (0-1): {v_measure_score(true_labels, pred_labels)}\n",
        f"Contigency Matrix: \n{contingency_matrix(true_labels, pred_labels)}\n",
        f"Purity: \n{purity_score(true_labels, pred_labels)}"]

def get_validation_df(true_labels, pred_labels):
    return pd.DataFrame([[
        rand_score(true_labels, pred_labels),
        homogeneity_score(true_labels, pred_labels),
        completeness_score(true_labels, pred_labels),
        v_measure_score(true_labels, pred_labels),
        purity_score(true_labels, pred_labels)
    ]])
        
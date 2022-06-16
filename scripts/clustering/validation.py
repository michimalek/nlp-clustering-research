from sklearn.metrics import rand_score, homogeneity_score, completeness_score, v_measure_score

def get_validation(true_labels, pred_labels):
    print(f"Rand Index (0-1): {rand_score(true_labels, pred_labels)}")
    print(f"Homogeneity Score: {homogeneity_score(true_labels, pred_labels)}")
    print(f"Completness Score: {completeness_score(true_labels, pred_labels)}")
    print(f"V-Measure (0-1): {v_measure_score(true_labels, pred_labels)}")
import numpy as np

def precision_at_k(actual, predicted, k=10):
    pred_k = predicted[:k]
    return len(set(pred_k) & set(actual)) / float(k)

def recall_at_k(actual, predicted, k=10):
    pred_k = predicted[:k]
    return len(set(pred_k) & set(actual)) / float(len(actual))

def ndcg_at_k(actual, predicted, k=10):
    dcg = sum([1 / np.log2(i+2) if predicted[i] in actual else 0 for i in range(min(k, len(predicted)))])
    idcg = sum([1 / np.log2(i+2) for i in range(min(k, len(actual)))])
    return dcg / idcg if idcg != 0 else 0.0

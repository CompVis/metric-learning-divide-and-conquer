import sklearn.cluster
import sklearn.metrics.cluster
import torch

def calc_normalized_mutual_information(ys, xs_clustered):
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys)

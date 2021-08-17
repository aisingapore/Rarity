import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics


def create_clusters(loss_list, num_cluster):
    """
    cluster creation for single model
    """
    loss_array = np.array(loss_list).reshape(-1, 1)
    kmean = KMeans(n_clusters=num_cluster, random_state=42).fit(loss_array)
    cluster_groups = kmean.labels_.tolist()
    cluster_groups = [cluster + 1 for cluster in cluster_groups]  # cluster to start from 1, instead of 0

    cluster_score = round(metrics.silhouette_score(loss_array, cluster_groups, metric='euclidean'), 4)
    return cluster_groups, cluster_score


def calculate_logloss(yTrue, yPred, log_func=math.log, eps=1e-15):
    """
    calculate single data point loss
    [sklearn] : Log loss is undefined for p=0 or p=1, so probabilities are clipped to max(eps, min(1 - eps, p))
    """
    yPred = np.clip(yPred, eps, 1 - eps)
    loss_single_datapoint = -(-yTrue * log_func(yPred) + (1 - yTrue) * log_func(1 - yPred))
    return loss_single_datapoint


def find_optimum_num_clusters(loss_list, num_cluster):
    loss_array = np.array(loss_list).reshape(-1, 1)
    cluster_range = list(range(1, 10))

    sum_squared_distance = []
    for i in cluster_range:
        kmean = KMeans(n_clusters=i, random_state=42).fit(loss_array)
        sum_squared_distance.append(kmean.inertia_)
    return cluster_range, sum_squared_distance

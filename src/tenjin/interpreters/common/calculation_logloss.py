# import numpy as np
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics


def create_clusters(loss_list, num_cluster):
    """
    cluster creation for single model
    """
    loss_array = np.array(loss_list).reshape(-1, 1)
    clusters = KMeans(n_clusters=num_cluster, random_state=42).fit(loss_array)
    cluster_groups = clusters.labels_.tolist()
    cluster_groups = [cluster + 1 for cluster in cluster_groups]  # cluster to start from 1, instead of 0

    cluster_score = round(metrics.silhouette_score(loss_array, cluster_groups, metric='euclidean'), 4)
    return cluster_groups, cluster_score

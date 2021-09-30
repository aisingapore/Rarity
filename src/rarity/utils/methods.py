# Copyright 2021 AI Singapore. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, List
import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances


def create_clusters(loss_list: List[float], num_cluster: int):
    '''
    cluster creation for single model
    '''
    loss_array = np.array(loss_list).reshape(-1, 1)

    kmean = KMeans(n_clusters=num_cluster, random_state=42).fit(loss_array)

    # this lookup index is to ensure the same clustering results (cluster groups) even if Kmean randomized the initalization at diff centroid
    # reference: https://stackoverflow.com/questions/44888415/how-to-set-k-means-clustering-labels-from-highest-to-lowest-with-python
    idx = np.argsort(kmean.cluster_centers_.sum(axis=1))
    lookup = np.zeros_like(idx)
    lookup[idx] = np.arange(num_cluster)
    cluster_groups = lookup[kmean.labels_].tolist()

    # cluster_groups = kmean.labels_.tolist() ==> use this if don't mind different clustering groups at each button click
    cluster_groups = [cluster + 1 for cluster in cluster_groups]  # cluster to start from 1, instead of 0
    cluster_score = round(metrics.silhouette_score(loss_array, cluster_groups, metric='euclidean'), 4)
    return cluster_groups, cluster_score


def calculate_logloss(yTrue: float, yPred: float, log_func: math.log = math.log, eps: float = 1e-15):
    '''
    calculate single data point loss
    [sklearn] : Log loss is undefined for p=0 or p=1, so probabilities are clipped to max(eps, min(1 - eps, p))
    '''
    yPred = np.clip(yPred, eps, 1 - eps)
    loss_single_datapoint = -(-yTrue * log_func(yPred) + (1 - yTrue) * log_func(1 - yPred))
    return loss_single_datapoint


def get_optimum_num_clusters(loss_list: List[float], num_cluster: int):
    '''
    generate 2 output lists for cluster count and its corresponding sum of squared distance
    '''
    loss_array = np.array(loss_list).reshape(-1, 1)
    cluster_range = list(range(1, 10))

    sum_squared_distance = []
    for i in cluster_range:
        kmean = KMeans(n_clusters=i, random_state=42).fit(loss_array)
        sum_squared_distance.append(kmean.inertia_)
    return cluster_range, sum_squared_distance


def calculate_kl_div(seq1: Union[pd.DataFrame, List[float]], seq2: Union[pd.DataFrame, List[float]]):
    kl_divergence = stats.entropy(seq1, seq2)
    return kl_divergence


def get_optimum_bin_size(feature_pd_series: pd.Series):
    '''
    adopt Freedman-Diaconis rule
    '''
    q1 = feature_pd_series.quantile(0.25)
    q3 = feature_pd_series.quantile(0.75)
    inner_quantile = q3 - q1
    bin_width = (2 * inner_quantile) / (len(feature_pd_series) ** (1 / 3))
    bin_count = int(np.ceil((feature_pd_series.max() - feature_pd_series.min()) / bin_width))
    return bin_count


def compute_distances(X: pd.Series, Y: pd.Series, metric=euclidean_distances):
    return metric(X, Y)

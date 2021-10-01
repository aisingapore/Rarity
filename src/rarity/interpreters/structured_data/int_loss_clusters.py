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

from typing import Union
import math

from rarity.data_loader import CSVDataLoader, DataframeLoader
from rarity.interpreters.structured_data.base_interpreters import BaseInterpreters
from rarity.utils.methods import create_clusters, calculate_logloss, get_optimum_num_clusters
from rarity.utils.common_functions import is_regression, is_classification


class IntLossClusterer(BaseInterpreters):
    '''
    Transform raw data into input format suitable for visualization on loss clusters

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module
    '''
    def __init__(self, data_loader: Union[CSVDataLoader, DataframeLoader]):
        super().__init__(data_loader)

    def extract_misspredictions(self):
        '''
        Function to tapout list of dataframe with prediction state info included
        '''
        ls_dfs_prob, _ = super().get_df_with_probability_values()
        ls_dfs_prob_misspred = [df.loc[lambda x: x['pred_state'] == 'miss-predict', :] for df in ls_dfs_prob]
        return ls_dfs_prob_misspred

    def xform(self, num_cluster: int, log_func: math.log, specific_dataset: str):
        '''
        Core transformation function to tap-out data into input format suitable for plotly graph

        Arguments:
            num_cluster (int):
                Number of cluster to form
            log_funct (:obj:`math.log`):
                Mathematics logarithm function used to calculate log-loss between yTrue and yPred
            specific_dataset (str):
                Default to 'All' indicating to include all miss-predict labels. Other options flexibly expand depending on class labels

        Returns:

                Compact outputs consist of the followings

                - df (:obj:`~pd.DataFrame`): dataframes for overview visualization need with offset values included
                - ls_score (:obj:`List[float]`): list of silhouette scores, indication of clustering quality
                - ls_cluster_range (:obj:`List[List[int]]`): list of list containing cluster number range from 1 to 10
                - ls_ssd (:obj:`List[float]`): sum of squared distance generated via kmean_inertia

        .. note::

            if classification, returns:

                Compact outputs consist of the followings

                - ls_dfs_viz (:obj:`List[~pd.DataFrame]`): dataframes for overview visualization need with offset values included
                - ls_class_labels (:obj:`List[str]`): list of all class labels
                - ls_class_labels_misspred (:obj:`List[str]`): list of class labels with minimum of 1 miss-prediction
                - ls_score (:obj:`List[float]`): list of silhouette scores, indication of clustering quality
                - ls_cluster_range (:obj:`List[List[int]]`): list of list containing cluster number range from 1 to 10
                - ls_ssd (:obj:`List[float]`): sum of squared distance generated via kmean_inertia
        '''
        if is_regression(self.analysis_type):
            df = super().get_df_with_offset_values()
            df.insert(0, 'index', df.index)  # for ease of user to trace the datapoint in raw dataframe

            cluster_groups_m1, cluster_score_m1 = create_clusters(df[f'offset_{self.models[0]}'], num_cluster)
            df[f'cluster_{self.models[0]}'] = cluster_groups_m1

            cluster_range_m1, sum_squared_distance_m1 = get_optimum_num_clusters(df[f'offset_{self.models[0]}'], num_cluster)
            # wrapped in list to std with classification output format for ease of feature integration at later stage
            ls_score = [cluster_score_m1]
            ls_cluster_range = [cluster_range_m1]
            ls_ssd = [sum_squared_distance_m1]

            if len(self.models) == 2:
                cluster_groups_m2, cluster_score_m2 = create_clusters(df[f'offset_{self.models[1]}'], num_cluster)
                df[f'cluster_{self.models[1]}'] = cluster_groups_m2

                cluster_range_m2, sum_squared_distance_m2 = get_optimum_num_clusters(df[f'offset_{self.models[1]}'], num_cluster)
                ls_score.append(cluster_score_m2)
                ls_cluster_range.append(cluster_range_m2)
                ls_ssd.append(sum_squared_distance_m2)

            return df, ls_score, ls_cluster_range, ls_ssd

        elif is_classification(self.analysis_type):
            ls_dfs_prob, ls_class_labels = super().get_df_with_probability_values()
            ls_dfs_prob_misspred = [df.loc[lambda x: x['pred_state'] == 'miss-predict', :] for df in ls_dfs_prob]
            ls_class_labels_misspred = [list(df['yTrue'].astype('str').unique()) for df in ls_dfs_prob_misspred]
            if len(ls_dfs_prob) == 2:
                ls_class_labels_misspred = list(set(ls_class_labels_misspred[0] + ls_class_labels_misspred[1]))
            else:
                ls_class_labels_misspred = ls_class_labels_misspred[0]

            try:
                ls_class_labels_misspred = [int(label) for label in ls_class_labels_misspred]
                ls_class_labels_misspred.sort()
            except TypeError:
                ls_class_labels_misspred.sort()

            ls_score = []
            ls_dfs_viz = []
            ls_cluster_range = []
            ls_ssd = []

            for df in ls_dfs_prob_misspred:
                df['eff_prob_for_loss_cal'] = [df[df['yTrue'].astype('str').values[i]].values[i] for i in range(len(df))]

                if specific_dataset != 'All':
                    try:
                        df_viz = df.loc[lambda x: x['yTrue'].astype('str') == specific_dataset, :]
                    except ValueError:
                        pass
                else:
                    df_viz = df

                df_temp = df_viz[['yTrue', 'eff_prob_for_loss_cal']]
                df_viz['lloss'] = df_temp.apply(lambda x: calculate_logloss(x['yTrue'], x['eff_prob_for_loss_cal'], log_func), axis=1)
                df_viz.pop('eff_prob_for_loss_cal')

                cluster_groups, cluster_score = create_clusters(df_viz['lloss'], num_cluster)
                df_viz['cluster'] = cluster_groups

                cluster_range, sum_squared_distance = get_optimum_num_clusters(df_viz['lloss'], num_cluster)

                ls_score.append(cluster_score)
                ls_dfs_viz.append(df_viz)
                ls_cluster_range.append(cluster_range)
                ls_ssd.append(sum_squared_distance)

            return ls_dfs_viz, ls_class_labels, ls_class_labels_misspred, ls_score, ls_cluster_range, ls_ssd

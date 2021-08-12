# import numpy as np
from tenjin.interpreters.structured_data.base_interpreters import BaseInterpreters
from tenjin.interpreters.common import create_clusters, calculate_logloss, find_optimum_num_clusters


class IntLossClusterer(BaseInterpreters):
    def __init__(self, data_loader):
        super().__init__(data_loader)

    def xform(self, num_cluster, log_func):
        if self.analysis_type == 'regression':
            df = super().get_df_with_offset_values()
            df.insert(0, 'index', df.index)  # for ease of user to trace the datapoint in raw dataframe

            cluster_groups_m1, cluster_score_m1 = create_clusters(df[f'offset_{self.models[0]}'], num_cluster)
            df[f'cluster_{self.models[0]}'] = cluster_groups_m1

            cluster_range_m1, sum_squared_distance_m1 = find_optimum_num_clusters(df[f'offset_{self.models[0]}'], num_cluster)
            ls_score = [cluster_score_m1]
            ls_cluster_range = [cluster_range_m1]
            ls_ssd = [sum_squared_distance_m1]

            if len(self.models) == 2:
                cluster_groups_m2, cluster_score_m2 = create_clusters(df[f'offset_{self.models[1]}'], num_cluster)
                df[f'cluster_{self.models[1]}'] = cluster_groups_m2

                cluster_range_m2, sum_squared_distance_m2 = find_optimum_num_clusters(df[f'offset_{self.models[1]}'], num_cluster)
                ls_score = [cluster_score_m1, cluster_score_m2]
                ls_cluster_range = [cluster_range_m1, cluster_range_m2]
                ls_ssd = [sum_squared_distance_m1, sum_squared_distance_m2]

            return df, ls_score, self.analysis_type, ls_cluster_range, ls_ssd

        elif 'classification' in self.analysis_type:
            ls_dfs_prob, ls_class_labels = super().get_df_with_probability_values()

            ls_score = []
            ls_dfs_viz = []
            ls_cluster_range = []
            ls_ssd = []
            if len(ls_class_labels) == 2:  # binary classification (supporting both single and bimodal)
                for df in ls_dfs_prob:
                    df_viz = df.loc[lambda x: x['pred_state'] == 'miss-predict', :]

                    df_temp = df_viz[['yTrue', ls_class_labels[1]]]
                    df_viz['lloss'] = df_temp.apply(lambda x: calculate_logloss(x['yTrue'], x[ls_class_labels[1]], log_func), axis=1)

                    cluster_groups, cluster_score = create_clusters(df_viz['lloss'], num_cluster)
                    df_viz['cluster'] = cluster_groups

                    cluster_range, sum_squared_distance = find_optimum_num_clusters(df_viz['lloss'], num_cluster)

                    ls_score.append(cluster_score)
                    ls_dfs_viz.append(df_viz)
                    ls_cluster_range.append(cluster_range)
                    ls_ssd.append(sum_squared_distance)

            return ls_dfs_viz, ls_class_labels, ls_score, self.analysis_type, ls_cluster_range, ls_ssd

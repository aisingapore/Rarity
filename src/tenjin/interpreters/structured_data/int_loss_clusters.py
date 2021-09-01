from tenjin.interpreters.structured_data.base_interpreters import BaseInterpreters
from tenjin.utils.methods import create_clusters, calculate_logloss, get_optimum_num_clusters
from tenjin.utils.common_functions import is_regression, is_classification


class IntLossClusterer(BaseInterpreters):
    def __init__(self, data_loader):
        super().__init__(data_loader)

    def extract_misspredictions(self):
        ls_dfs_prob, _ = super().get_df_with_probability_values()
        ls_dfs_prob_misspred = [df.loc[lambda x: x['pred_state'] == 'miss-predict', :] for df in ls_dfs_prob]
        return ls_dfs_prob_misspred

    def xform(self, num_cluster, log_func, specific_dataset):
        if is_regression(self.analysis_type):
            df = super().get_df_with_offset_values()
            df.insert(0, 'index', df.index)  # for ease of user to trace the datapoint in raw dataframe

            cluster_groups_m1, cluster_score_m1 = create_clusters(df[f'offset_{self.models[0]}'], num_cluster)
            df[f'cluster_{self.models[0]}'] = cluster_groups_m1

            cluster_range_m1, sum_squared_distance_m1 = get_optimum_num_clusters(df[f'offset_{self.models[0]}'], num_cluster)
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

            return df, ls_score, self.analysis_type, ls_cluster_range, ls_ssd

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
                        pass  # # **** need to check error at this point
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

            return ls_dfs_viz, ls_class_labels, ls_class_labels_misspred, ls_score, self.analysis_type, ls_cluster_range, ls_ssd

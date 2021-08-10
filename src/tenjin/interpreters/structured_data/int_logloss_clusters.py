from tenjin.interpreters.structured_data.base_interpreters import BaseInterpreters
from tenjin.interpreters.common import create_clusters


class IntLossClusterer(BaseInterpreters):
    def __init__(self, data_loader):
        super().__init__(data_loader)

    def xform(self, num_cluster):
        if self.analysis_type == 'regression':
            df = super().get_df_with_offset_values()
            df.insert(0, 'index', df.index)  # for ease of user to trace the datapoint in raw dataframe

            cluster_groups_m1, cluster_score_m1 = create_clusters(df[f'offset_{self.models[0]}'], num_cluster)
            df[f'cluster_{self.models[0]}'] = cluster_groups_m1
            score_list = [cluster_score_m1]

            if len(self.models) == 2:
                cluster_groups_m2, cluster_score_m2 = create_clusters(df[f'offset_{self.models[1]}'], num_cluster)
                df[f'cluster_{self.models[1]}'] = cluster_groups_m2
                score_list = [cluster_score_m1, cluster_score_m2]
            return df, score_list

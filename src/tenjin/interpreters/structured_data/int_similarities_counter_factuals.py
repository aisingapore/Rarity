# from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tenjin.interpreters.structured_data.base_interpreters import BaseInterpreters
# from tenjin.interpreters.common.calculate_counterfactuals import compute_distances
from tenjin.utils.methods import compute_distances
from tenjin.utils.common_functions import is_regression, is_classification, insert_index_col


class IntSimilaritiesCounterFactuals(BaseInterpreters):
    # def __init__(self, data_loader, user_defined_idx, top_n_CF=3):
    def __init__(self, data_loader):
        super().__init__(data_loader)
        self.df_features = data_loader.get_features()
        # self.user_defined_idx = user_defined_idx
        # self.top_n_CF = top_n_CF

    def _get_categorical_x_feature(self, df):
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        return categorical_cols

    def _label_encode_categorical_features(self, df, categorical_cols):
        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[col] = LabelEncoder.fit_transform(df_encoded[col])
        return df_encoded

    def _apply_standard_scale(self, df):
        scaler = StandardScaler()
        return scaler.fit_transform(df)

    def xform(self, user_defined_idx, feature_to_exclude=[], top_n_CF=3):
        if is_regression(self.analysis_type):
            df_viz = super().get_df_with_offset_values()
            df_viz = insert_index_col(df_viz)

            categorical_cols = self._get_categorical_x_feature(self.df_features)
            df_encoded = self._label_encode_categorical_features(self.df_features, categorical_cols)
            # print(f'df_encoded: {df_encoded}')
            scaled_data = self._apply_standard_scale(df_encoded)
            df_scaled = pd.DataFrame(scaled_data, index=df_encoded.index, columns=df_encoded.columns)
            # df_scaled = insert_index_col(df_scaled)
            # print(f'df_scaled: {df_scaled}')
            baseline = df_scaled.iloc[[user_defined_idx]]
            # df_base = df_viz.loc[lambda x: x['index'] == user_defined_idx, :]
            # df_base.insert(1, 'calculated_distance', 0)
            # df_base = df_viz.iloc[[user_defined_idx]]

            distance_metrics_dict = {}
            for idx in df_scaled.index:
                # if idx != user_defined_idx:
                distance_metrics_dict[idx] = compute_distances(baseline, df_scaled.loc[lambda x: x.index == idx, :])[0][0]

            sorted_distance_metrics_dict = dict(sorted(distance_metrics_dict.items(), key=lambda x: x[1]))
            # print(f'len sorted dis-metrics-dict: {len(sorted_distance_metrics_dict)}')
            # print(f'sorted distance-metrics-dict: {sorted_distance_metrics_dict}')

            idx_for_top_n_CF = [k for k in list(sorted_distance_metrics_dict.keys())[:top_n_CF + 1]]  # top_n_CF + 1 => first is user_defined_idx
            calculated_distance = [round(v, 4) for v in list(sorted_distance_metrics_dict.values())[:top_n_CF + 1]]

            df_top_n_CF = df_viz.loc[lambda x: x['index'].isin(idx_for_top_n_CF), :]
            df_top_n_CF.insert(1, 'calculated_distance', calculated_distance)
            # df_final_viz = pd.concat([df_base, df_top_n_CF], axis=0)

        elif is_classification:
            pass
        # return df_final_viz
        return df_top_n_CF

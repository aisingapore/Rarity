import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tenjin.interpreters.structured_data.base_interpreters import BaseInterpreters
from tenjin.utils.methods import compute_distances
from tenjin.utils.common_functions import is_regression, is_classification, insert_index_col


class IntSimilaritiesCounterFactuals(BaseInterpreters):
    def __init__(self, data_loader):
        super().__init__(data_loader)
        self.df_features = data_loader.get_features()
        self.models = data_loader.get_model_list()

    def _get_categorical_features(self, df):
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        return categorical_cols

    def _label_encode_categorical_features(self, df, categorical_cols):
        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
        return df_encoded

    def _apply_standard_scale(self, df):
        scaler = StandardScaler()
        return scaler.fit_transform(df)

    def _get_df_sorted_by_distance_metrics(self, user_defined_idx, feature_to_exclude, top_n):
        categorical_cols = self._get_categorical_features(self.df_features)
        df_encoded = self._label_encode_categorical_features(self.df_features, categorical_cols)
        scaled_data = self._apply_standard_scale(df_encoded)
        df_scaled = pd.DataFrame(scaled_data, index=df_encoded.index, columns=df_encoded.columns)
        df_scaled = df_scaled[[col for col in df_scaled.columns if col not in feature_to_exclude]]
        baseline = df_scaled.iloc[[user_defined_idx]]

        distance_metrics_dict = {}
        for idx in df_scaled.index:
            distance_metrics_dict[idx] = compute_distances(baseline, df_scaled.loc[lambda x: x.index == idx, :])[0][0]

        sorted_distance_metrics_dict = dict(sorted(distance_metrics_dict.items(), key=lambda x: x[1]))

        idx_for_top_n = [k for k in list(sorted_distance_metrics_dict.keys())[:top_n + 1]]  # top_n + 1 => first is user_defined_idx
        calculated_distance = [round(v, 4) for v in list(sorted_distance_metrics_dict.values())[:top_n + 1]]
        return idx_for_top_n, calculated_distance

    def xform(self, user_defined_idx, feature_to_exclude=None, top_n=3):
        if not isinstance(feature_to_exclude, list):
            try:
                feature_to_exclude = list(feature_to_exclude)
            except TypeError:  # 'NoneType' object is not iterable
                feature_to_exclude = []

        if is_regression(self.analysis_type):
            df_viz = super().get_df_with_offset_values()
            df_viz = insert_index_col(df_viz)

            idx_for_top_n, calculated_distance = self._get_df_sorted_by_distance_metrics(user_defined_idx, feature_to_exclude, top_n)

        elif is_classification(self.analysis_type):
            df_features = insert_index_col(self.df_features)
            yTrue = insert_index_col(self.data_loader.get_yTrue())
            df_viz = df_features.merge(yTrue, how='left', on='index')

            yPreds = self.data_loader.get_yPreds()
            for i, model in enumerate(self.models):
                df_viz[f'yPred_{model}'] = yPreds[i]['yPred-label']

            idx_for_top_n, calculated_distance = self._get_df_sorted_by_distance_metrics(user_defined_idx, feature_to_exclude, top_n)
        return df_viz, idx_for_top_n, calculated_distance

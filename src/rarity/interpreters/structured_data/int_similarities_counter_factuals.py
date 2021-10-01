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

from typing import Union, List, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from rarity.data_loader import CSVDataLoader, DataframeLoader
from rarity.interpreters.structured_data.base_interpreters import BaseInterpreters
from rarity.utils.methods import compute_distances
from rarity.utils.common_functions import is_regression, is_classification, insert_index_col


class IntSimilaritiesCounterFactuals(BaseInterpreters):
    '''
    Transform raw data into input format suitable for visualization on Similarities / Counter-Factuals

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module
    '''
    def __init__(self, data_loader: Union[CSVDataLoader, DataframeLoader]):
        super().__init__(data_loader)
        self.df_features = data_loader.get_features()
        self.models = data_loader.get_model_list()

    def _get_categorical_features(self, df: pd.DataFrame):
        '''
        Identify categorical features
        '''
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        return categorical_cols

    def _label_encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List):
        '''
        Fit-transform categorical features with ``LabelEncoder``
        '''
        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
        return df_encoded

    def _apply_standard_scale(self, df: pd.DataFrame):
        '''
        Standard scale features
        '''
        scaler = StandardScaler()
        return scaler.fit_transform(df)

    def _get_ranking_and_distance_metrics(self, user_defined_idx: int, feature_to_exclude: List, top_n: int):
        '''
        Compute distance scores and generate index list sorted by distance ranking
        '''
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

    def xform(self, user_defined_idx: int, feature_to_exclude: Optional[List[str]] = None, top_n: int = 3):
        '''
        Core transformation function to tap-out data into input format suitable for plotly graph

        Arguments:
            user_defined_idx (int):
                Index of the data point of interest specified by user
            feature_to_exclude (List of :obj:`str`, `optional`):
                A list of features to be excluded from the ranking and similarities distance calculation
            top_n (int):
                Number indicating the max limit of records to be displayed based on the distance ranking

        Returns:

                Outputs consist of the followings

                - idx_for_top_n (:obj:`List[int]`): list of integer numbers indicating the ranking position in ascending order
                - calculated_distance (:obj:`List[float]`): list of calculated euclidean_distances

        .. note::

            if classification, returns:

                Outputs consist of the followings

                - df_viz (:obj:`~pd.DataFrame`): dataframes for overview visualization need with true labels and predicted labels included
                - idx_for_top_n (:obj:`List[int]`): list of integer numbers indicating the ranking position in ascending order
                - calculated_distance (:obj:`List[float]`): list of calculated euclidean_distances

        '''
        if not isinstance(feature_to_exclude, list):
            try:
                feature_to_exclude = list(feature_to_exclude)
            except TypeError:  # 'NoneType' object is not iterable
                feature_to_exclude = []

        if is_regression(self.analysis_type):
            df_viz = super().get_df_with_offset_values()
            df_viz = insert_index_col(df_viz)

            idx_for_top_n, calculated_distance = self._get_ranking_and_distance_metrics(user_defined_idx, feature_to_exclude, top_n)

        elif is_classification(self.analysis_type):
            df_features = insert_index_col(self.df_features)
            yTrue = insert_index_col(self.data_loader.get_yTrue())
            df_viz = df_features.merge(yTrue, how='left', on='index')

            yPreds = self.data_loader.get_yPreds()
            for i, model in enumerate(self.models):
                df_viz[f'yPred_{model}'] = yPreds[i]['yPred-label']

            idx_for_top_n, calculated_distance = self._get_ranking_and_distance_metrics(user_defined_idx, feature_to_exclude, top_n)
        return df_viz, idx_for_top_n, calculated_distance

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
import numpy as np
import pandas as pd

from rarity.data_loader import CSVDataLoader, DataframeLoader
from rarity.interpreters.structured_data.base_interpreters import BaseInterpreters
from rarity.utils.methods import calculate_kl_div, get_optimum_bin_size
from rarity.utils.common_functions import insert_index_col, is_regression, is_classification


class IntFeatureDistribution(BaseInterpreters):
    '''
    Transform raw data into input format suitable for visualization on feature distribution

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module
    '''
    def __init__(self, data_loader: Union[CSVDataLoader, DataframeLoader]):
        super().__init__(data_loader)
        self.df_features = insert_index_col(self.data_loader.get_features())
        # default range set as last 20% of dataset sample size
        self.df_default_range = self.df_features.iloc[np.r_[int(len(self.df_features) * 0.8):, :]]
        self.features = self.df_features.columns

    def _get_df_sliced(self, start_idx: int, stop_idx: int):
        '''
        Slice dataframe to the specific range.
        '''
        if start_idx is not None and stop_idx is None:
            df_sliced = self.df_features.iloc[np.r_[start_idx:, :]]
        elif start_idx is None and stop_idx is not None:
            df_sliced = self.df_features.iloc[np.r_[:stop_idx, :]]
        elif start_idx is not None and stop_idx is not None:
            df_sliced = self.df_features.iloc[np.r_[start_idx:stop_idx, :]]
        else:  # range is not specified
            df_sliced = self.df_features
            if is_regression(self.analysis_type):
                df_sliced = self.df_default_range
        return df_sliced

    def _get_single_feature_df_with_binning(self, df: pd.DataFrame, feature: str):
        '''
        For regression task only.
        Function to find optimum bin-size on sliced df for distribution comparison
        '''
        df_viz_specific_feat = df[['dataset_type', feature]]
        optimum_bin_size = get_optimum_bin_size(df_viz_specific_feat[feature])
        df_viz_specific_feat['bin_group'] = pd.cut(df_viz_specific_feat[feature], optimum_bin_size, labels=list(range(optimum_bin_size)))
        return df_viz_specific_feat, optimum_bin_size

    def _get_probabilities_by_bin_group(self, df_viz: pd.DataFrame, bin_count: int):
        '''
        For regression task only.
        Function to tap-out customized df for ease of getting probabilities based on bin group for reference df and sliced df
        '''
        def _interim_df_xformed_from_bin_group(specific_pd_series: pd.Series, col_name: str):
            interim_df = specific_pd_series.value_counts().rename_axis('bin_group').reset_index(name=col_name).sort_values('bin_group')
            return interim_df

        df_probs_by_bin = pd.DataFrame()
        df_probs_by_bin['bin_group'] = [n for n in range(bin_count)]
        df_filter_ref = df_viz.loc[lambda x:x['dataset_type'] == 'df_reference', :]
        df_filter_sliced = df_viz.loc[lambda x:x['dataset_type'] == 'df_sliced', :]

        df_temp_ref = _interim_df_xformed_from_bin_group(df_filter_ref['bin_group'], 'df_ref_counts')
        df_temp_sliced = _interim_df_xformed_from_bin_group(df_filter_sliced['bin_group'], 'df_sliced_counts')

        df_kl_div = df_probs_by_bin.merge(df_temp_ref, how='left', on='bin_group')
        df_kl_div = df_kl_div.merge(df_temp_sliced, how='left', on='bin_group')
        df_kl_div.fillna(0, inplace=True)
        df_kl_div['df_ref_counts'] = df_kl_div['df_ref_counts'].replace(0, 1e-15)  # to avoid division by zero
        df_kl_div['df_sliced_counts'] = df_kl_div['df_sliced_counts'].replace(0, 1e-15)  # to avoid division by zero
        df_kl_div['df_ref_counts_pct'] = [v / sum(df_kl_div['df_ref_counts'].values) for v in df_kl_div['df_ref_counts'].values]
        df_kl_div['df_sliced_counts_pct'] = [v / sum(df_kl_div['df_sliced_counts'].values) for v in df_kl_div['df_sliced_counts'].values]

        # keep in dataframe format for ease of debugging / troubleshooting
        probs_df_ref = df_kl_div['df_ref_counts_pct']
        probs_df_sliced = df_kl_div['df_sliced_counts_pct']
        return probs_df_ref, probs_df_sliced

    def _get_df_feature_with_pred_state_cls(self, df_overall: pd.DataFrame):
        '''
        For classification task only.
        Function to tap-out customized df combining features and relevant prediction info for use in visualization.
        '''
        ls_dfs_viz, _ = super().get_df_with_probability_values()

        ls_dfs_viz_featdist = []
        for df_viz in ls_dfs_viz:
            df_viz = insert_index_col(df_viz)
            df_predstate = df_viz[['index', 'yTrue', 'yPred-label', 'pred_state', 'model']]
            df_viz_interim = df_overall.merge(df_predstate, how='left', on='index')
            ls_dfs_viz_featdist.append(df_viz_interim)
        return ls_dfs_viz_featdist

    def _get_probabilities_by_feature(self, df_viz: pd.DataFrame, specific_feature: str):
        '''
        For classification task only.
        Function to calculate probabilities of correct vs miss-predict for specific feature
        '''
        df_pivot = pd.pivot_table(
            df_viz[[specific_feature, 'pred_state', 'model']],
            index=specific_feature,
            values='model',
            columns='pred_state',
            aggfunc='count',
            fill_value=1e-15)  # to avoid inf due to zero division for those NA or zero

        # set up new columns to get the percentage of each subvalue of x_colName
        df_pivot['correct'] = df_pivot['correct'].replace(0, 1e-15)  # to avoid division by zero
        df_pivot['miss-predict'] = df_pivot['miss-predict'].replace(0, 1e-15)  # to avoid division by zero
        df_pivot['correct_pct'] = [v / sum(df_pivot['correct']) for v in df_pivot['correct']]
        df_pivot['misspredict_pct'] = [v / sum(df_pivot['miss-predict']) for v in df_pivot['miss-predict']]

        probs_correct = df_pivot['correct_pct']
        probs_misspred = df_pivot['misspredict_pct']
        return probs_correct, probs_misspred

    def _generate_kl_div_info_base(self, df: pd.DataFrame, feature_to_exclude: List):
        '''
        Function to generate dictionary like output storing kl-divergence score for each feature
        arranged in descending order.
        '''
        kl_div_dict = {}
        for feat in self.features:
            if feat not in feature_to_exclude:
                if is_regression(self.analysis_type):
                    df_viz_specific_feat, optimum_bin_size = self._get_single_feature_df_with_binning(df, feat)
                    probs_df_ref, probs_df_sliced = self._get_probabilities_by_bin_group(df_viz_specific_feat, optimum_bin_size)
                    kl_div = calculate_kl_div(probs_df_ref, probs_df_sliced)
                    kl_div_dict[feat] = [kl_div, df_viz_specific_feat]

                elif is_classification(self.analysis_type):
                    df_viz_specific_feat = df[[feat, 'pred_state', 'model']]
                    try:
                        probs_correct, probs_misspred = self._get_probabilities_by_feature(df, feat)
                        kl_div = calculate_kl_div(probs_correct, probs_misspred)
                    except KeyError:  # KeyError: 'miss-predict' => no miss-predict for the selected idx range
                        kl_div = 0  # no comparison is feasible, therefore divergence is 0
                    kl_div_dict[feat] = [kl_div, df_viz_specific_feat]

        kl_div_dict_sorted = dict(sorted(kl_div_dict.items(), key=lambda x: x[1][0], reverse=True))
        return kl_div_dict_sorted

    def xform(self, feature_to_exclude: Optional[List[str]] = None, start_idx: Optional[int] = None, stop_idx: Optional[int] = None):
        '''
        Core transformation function to tap-out data into input format suitable for plotly graph

        Arguments:
            feature_to_exclude (List of :obj:`str`, `optional`):
                A list of features to be excluded from the kl-div calculation and visualization
            start_idx (:obj:`int`, `optional`):
                Integer number indicating the start index position to slice dataframe
            stop_idx (:obj:`int`, `optional`):
                Integer number indicating the stop index position to slice dataframe

        Returns:
            :obj:`Dict` or :obj:`List(Dict)`:
                dictionary storing kl-divergence score for each feature in decending order
        '''
        if isinstance(feature_to_exclude, list):
            feature_to_exclude = feature_to_exclude
        else:
            try:
                feature_to_exclude = [feature_to_exclude]
            except TypeError:
                feature_to_exclude = []

        df_sliced = self._get_df_sliced(start_idx, stop_idx)
        idx_sliced_df = list(df_sliced.index)

        df_overall = self.df_features.copy()
        df_overall['dataset_type'] = ['df_sliced' if idx in idx_sliced_df else 'df_reference' for idx in df_overall.index]

        if is_regression(self.analysis_type):
            kl_div_dict_sorted = self._generate_kl_div_info_base(df_overall, feature_to_exclude)
            kl_div_dict_sorted.pop('index')
            return kl_div_dict_sorted

        elif is_classification(self.analysis_type):
            if start_idx is not None and stop_idx is not None:  # user has specified a slicing range to inspect
                df_overall = df_overall.loc[lambda x: x['dataset_type'] == 'df_sliced', :]

            ls_dfs_viz_featdist = self._get_df_feature_with_pred_state_cls(df_overall)

            ls_kl_div_dict_sorted = []
            for df_viz in ls_dfs_viz_featdist:
                kl_div_dict_sorted = self._generate_kl_div_info_base(df_viz, feature_to_exclude)
                kl_div_dict_sorted.pop('index')
                ls_kl_div_dict_sorted.append(kl_div_dict_sorted)
        return ls_kl_div_dict_sorted

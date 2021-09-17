import numpy as np
import pandas as pd

from tenjin.interpreters.structured_data.base_interpreters import BaseInterpreters
from tenjin.interpreters.common import calculate_kl_div, get_optimum_bin_size
from tenjin.utils.common_functions import insert_index_col


class IntFeatureDistribution(BaseInterpreters):
    def __init__(self, data_loader):
        super().__init__(data_loader)
        self.df_features = insert_index_col(self.data_loader.get_features())
        # default range set as last 20% of dataset sample size
        self.df_default_range = self.df_features.iloc[np.r_[int(len(self.df_features) * 0.8):, :]]
        self.features = self.df_features.columns

    def _get_df_sliced(self, start_idx, stop_idx):
        if start_idx is not None and stop_idx is None:
            df_sliced = self.df_features.iloc[np.r_[start_idx:, :]]
        elif start_idx is None and stop_idx is not None:
            df_sliced = self.df_features.iloc[np.r_[:stop_idx, :]]
        elif start_idx is not None and stop_idx is not None:
            df_sliced = self.df_features.iloc[np.r_[start_idx:stop_idx, :]]
        else:  # range is not specified
            if self.analysis_type == 'regression':
                df_sliced = self.df_default_range
            else:
                df_sliced = self.df_features
        return df_sliced

    def _get_single_feature_df_with_binning(self, df, feature):
        '''
        For regression task
        '''
        df_viz_specific_feat = df[['dataset_type', feature]]
        optimum_bin_size = get_optimum_bin_size(df_viz_specific_feat[feature])
        df_viz_specific_feat['bin_group'] = pd.cut(df_viz_specific_feat[feature], optimum_bin_size, labels=list(range(optimum_bin_size)))
        return df_viz_specific_feat, optimum_bin_size

    def _get_probabilities_by_bin_group(self, df_viz, bin_count):
        '''
        For regression task
        '''
        def _interim_df_xformed_from_bin_group(specific_pd_series, col_name):
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

    def _get_df_feature_with_pred_state_cls(self, df_overall):
        '''
        For classification task
        '''
        ls_dfs_viz, _ = super().get_df_with_probability_values()

        ls_dfs_viz_featdist = []
        for df_viz in ls_dfs_viz:
            df_viz = insert_index_col(df_viz)
            df_predstate = df_viz[['index', 'yTrue', 'yPred-label', 'pred_state', 'model']]
            df_viz_interim = df_overall.merge(df_predstate, how='left', on='index')
            ls_dfs_viz_featdist.append(df_viz_interim)
        return ls_dfs_viz_featdist

    def _get_probabilities_by_feature(self, df_viz, specific_feature):
        '''
        For classification task
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

    def _generate_kl_div_info_base(self, df, feature_to_exclude):
        kl_div_dict = {}
        for feat in self.features:
            if feat not in feature_to_exclude:
                if self.analysis_type == 'regression':
                    df_viz_specific_feat, optimum_bin_size = self._get_single_feature_df_with_binning(df, feat)
                    probs_df_ref, probs_df_sliced = self._get_probabilities_by_bin_group(df_viz_specific_feat, optimum_bin_size)
                    kl_div = calculate_kl_div(probs_df_ref, probs_df_sliced)
                    kl_div_dict[feat] = [kl_div, df_viz_specific_feat]

                elif 'classification' in self.analysis_type:
                    df_viz_specific_feat = df[[feat, 'pred_state', 'model']]
                    try:
                        probs_correct, probs_misspred = self._get_probabilities_by_feature(df, feat)
                        kl_div = calculate_kl_div(probs_correct, probs_misspred)
                    except KeyError:  # KeyError: 'miss-predict' => no miss-predict for the selected idx range
                        kl_div = 0  # no comparison is feasible, therefore divergence is 0
                    kl_div_dict[feat] = [kl_div, df_viz_specific_feat]

        kl_div_dict_sorted = dict(sorted(kl_div_dict.items(), key=lambda x: x[1][0], reverse=True))
        return kl_div_dict_sorted

    def xform(self, feature_to_exclude=None, start_idx=None, stop_idx=None):
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

        if self.analysis_type == 'regression':
            kl_div_dict_sorted = self._generate_kl_div_info_base(df_overall, feature_to_exclude)
            kl_div_dict_sorted.pop('index')
            return kl_div_dict_sorted

        elif 'classification' in self.analysis_type:
            if start_idx is not None and stop_idx is not None:  # user has specified a slicing range to inspect
                df_overall = df_overall.loc[lambda x: x['dataset_type'] == 'df_sliced', :]

            ls_dfs_viz_featdist = self._get_df_feature_with_pred_state_cls(df_overall)

            ls_kl_div_dict_sorted = []
            for df_viz in ls_dfs_viz_featdist:
                kl_div_dict_sorted = self._generate_kl_div_info_base(df_viz, feature_to_exclude)
                kl_div_dict_sorted.pop('index')
                ls_kl_div_dict_sorted.append(kl_div_dict_sorted)
        return ls_kl_div_dict_sorted
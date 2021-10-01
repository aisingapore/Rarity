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

from typing import Dict, List
import string
import pandas as pd


def identify_active_trace(relayout_data: Dict):
    '''
    only 1 unique trace should be in relayout_data
    this function is to retrieve the position info of the unique trace
    '''
    for idx, rdata in enumerate(relayout_data):
        if rdata is not None:
            if is_active_trace(rdata):
                specific_layout = idx, rdata
        else:
            continue
    return specific_layout


def is_active_trace(relayout_data: Dict):
    if ('xaxis.range[0]' in relayout_data.keys() or 'yaxis.range[0]' in relayout_data.keys()) and 'xaxis.autorange' not in relayout_data.keys():
        return True
    else:
        return False


def is_reset(relayout_data: Dict):
    if 'xaxis.autorange' in relayout_data.keys():
        return True
    else:
        return False


def is_regression(analysis_type: str):
    if analysis_type == 'regression':
        return True
    else:
        return False


def is_classification(analysis_type: str):
    if 'classification' in analysis_type:
        return True
    else:
        return False


def detected_bimodal(models: List[str]):
    if len(models) == 2:
        return True
    else:
        return False


def detected_invalid_index_inputs(specific_idx: str, df: pd.DataFrame):
    special_characters = list(string.punctuation.replace(',', ''))
    letters = list(string.ascii_letters)
    if any(sp_char for sp_char in special_characters if sp_char in specific_idx):
        return True
    elif any(letter for letter in letters if letter in specific_idx):
        return True
    elif any(idx for idx in specific_idx.replace(' ', '').split(',') if (int(idx) > len(df)) or (int(idx) < 0)):
        return True


def detected_legend_filtration(restyle_data: Dict):
    if restyle_data[0]['visible'] == ['legendonly']:
        return True
    elif restyle_data[0]['visible'] == [True]:
        return False
    else:
        return False


def detected_unique_figure(relayout_data: Dict):
    if sum(_valid_fig_key_ls(relayout_data)) == 1:
        return True
    else:
        return False


def detected_more_than_1_unique_figure(relayout_data: Dict):
    if sum(_valid_fig_key_ls(relayout_data)) > 1:
        return True
    else:
        return False


def detected_single_xaxis(relayout_data: Dict):
    if 'yaxis.range[0]' not in relayout_data.keys():
        return True
    else:
        return False


def detected_single_yaxis(relayout_data: Dict):
    if 'xaxis.range[0]' not in relayout_data.keys():
        return True
    else:
        return False


def detected_complete_pair_xaxis_yaxis(relayout_data: Dict):
    if 'xaxis.range[0]' in relayout_data.keys() and 'yaxis.range[0]' in relayout_data.keys():
        return True
    else:
        return False


def get_min_max_offset(df: pd.DataFrame, models: List[str]):
    min_offset = df[f'offset_{models[0]}'].min()
    max_offset = df[f'offset_{models[0]}'].max()

    if len(models) == 2:
        min_offset_m2 = df[f'offset_{models[1]}'].min()
        min_offset = min(min_offset, min_offset_m2)

        max_offset_m2 = df[f'offset_{models[1]}'].max()
        max_offset = max(max_offset, max_offset_m2)
    return min_offset, max_offset


def get_min_max_index(df: pd.DataFrame, models: List[str], y_start_idx: int, y_stop_idx: int):
    condition_y_min = (df[f'offset_{models[0]}'] >= y_start_idx)
    condition_y_max = (df[f'offset_{models[0]}'] <= y_stop_idx)
    df_temp = conditional_sliced_df(df, condition_y_min, condition_y_max)
    min_index = min(list(df_temp.index))
    max_index = max(list(df_temp.index))

    if len(models) == 2:
        condition_y_min_m2 = (df[f'offset_{models[1]}'] >= y_start_idx)
        condition_y_max_m2 = (df[f'offset_{models[1]}'] <= y_stop_idx)
        df_temp_m2 = conditional_sliced_df(df, condition_y_min_m2, condition_y_max_m2)
        min_index_m2 = min(list(df_temp_m2.index))
        min_index = min(min_index, min_index_m2)

        max_index_m2 = max(list(df_temp_m2.index))
        max_index = max(max_index, max_index_m2)
    return min_index, max_index


def get_min_max_cluster(df: pd.DataFrame, models: List[str], y_start_idx: int, y_stop_idx: int):
    condition_y_min = (df[f'offset_{models[0]}'] >= y_start_idx)
    condition_y_max = (df[f'offset_{models[0]}'] <= y_stop_idx)
    df_temp = conditional_sliced_df(df, condition_y_min, condition_y_max)
    min_cluster = df_temp[f'cluster_{models[0]}'].min()
    max_cluster = df_temp[f'cluster_{models[0]}'].max()

    if len(models) == 2:
        condition_y_min_m2 = (df[f'offset_{models[1]}'] >= y_start_idx)
        condition_y_max_m2 = (df[f'offset_{models[1]}'] <= y_stop_idx)
        df_temp_m2 = conditional_sliced_df(df, condition_y_min_m2, condition_y_max_m2)
        min_cluster_m2 = df_temp_m2[f'cluster_{models[1]}'].min()
        min_cluster = min(min_cluster, min_cluster_m2)

        max_cluster_m2 = df_temp_m2[f'cluster_{models[1]}'].max()
        max_cluster = max(max_cluster, max_cluster_m2)
    return min_cluster, max_cluster


def get_effective_xaxis_cluster(relayout_data: Dict):
    if relayout_data['xaxis.range[0]'] > 1 and relayout_data['xaxis.range[0]'] < 8:
        x_cluster = int(relayout_data['xaxis.range[0]']) + 1
    elif relayout_data['xaxis.range[0]'] <= 1:
        x_cluster = 1
    elif relayout_data['xaxis.range[0]'] >= 8:
        x_cluster = 8
    return x_cluster


def get_adjusted_dfs_based_on_legend_filtration(dfs: List[pd.DataFrame], models: List[str]):
    if models[0] != dfs[0]['model'].values[0]:
        dfs[0] = dfs[1]
    return dfs


def get_adjusted_xy_coordinate(relayout_data: Dict, df: pd.DataFrame):
    if detected_single_xaxis(relayout_data):
        x_start_idx = int(relayout_data['xaxis.range[0]']) if relayout_data['xaxis.range[0]'] >= 0 else 0
        x_stop_idx = int(relayout_data['xaxis.range[1]']) if relayout_data['xaxis.range[1]'] <= len(df) - 1 else len(df) - 1
        y_start_idx = 0
        y_stop_idx = 1

    elif detected_single_yaxis(relayout_data):
        x_start_idx = 0
        x_stop_idx = len(df) - 1
        y_start_idx = relayout_data['yaxis.range[0]'] if relayout_data['yaxis.range[0]'] >= 0 else 0
        y_stop_idx = relayout_data['yaxis.range[1]'] if relayout_data['yaxis.range[1]'] <= 1 else 1

    elif detected_complete_pair_xaxis_yaxis(relayout_data):
        x_start_idx = int(relayout_data['xaxis.range[0]']) if relayout_data['xaxis.range[0]'] >= 0 else 0
        x_stop_idx = int(relayout_data['xaxis.range[1]']) if relayout_data['xaxis.range[1]'] <= len(df) - 1 else len(df) - 1
        y_start_idx = relayout_data['yaxis.range[0]'] if relayout_data['yaxis.range[0]'] > 0 else 0
        y_stop_idx = relayout_data['yaxis.range[1]'] if relayout_data['yaxis.range[1]'] < 1 else 1
    return x_start_idx, x_stop_idx, y_start_idx, y_stop_idx


def get_max_value_on_slider(df: pd.DataFrame, component: str):
    if component == 'feat-dist':
        max_value = len(list(df.columns))
        if len(list(df.columns)) > 10:
            max_value = 10
        return max_value
    elif component == 'similaritiesCF':
        max_value = len(df)
        if len(df) > 10:
            max_value = 10
        return max_value


def dataframe_prep_on_model_count_by_yaxis_slice(df: pd.DataFrame, models: List[str], y_start_idx: int, y_stop_idx: int):
    '''
    For regression task
    '''
    offset_cols = [col for col in df.columns if 'offset_' in col]
    condition_m1_1 = (df[offset_cols[0]] >= y_start_idx)
    condition_m1_2 = (df[offset_cols[0]] <= y_stop_idx)

    if len(models) == 2:
        condition_m2_1 = (df[offset_cols[1]] >= y_start_idx)
        condition_m2_2 = (df[offset_cols[1]] <= y_stop_idx)
        df_final_m1 = conditional_sliced_df(df, condition_m1_1, condition_m1_2)
        df_final_m2 = conditional_sliced_df(df, condition_m2_1, condition_m2_2)

        final_filtered_idx = set(df_final_m1.index).union(set(df_final_m2.index))
        df_final = df[df['index'].isin(list(final_filtered_idx))]
    else:
        if offset_cols[0].replace('offset_', '') == models[0]:
            df_final = conditional_sliced_df(df, condition_m1_1, condition_m1_2)
        else:
            condition_m2_1 = (df[offset_cols[1]] >= y_start_idx)
            condition_m2_2 = (df[offset_cols[1]] <= y_stop_idx)
            df_final = conditional_sliced_df(df, condition_m2_1, condition_m2_2)
    return df_final


def new_dataframe_prep_based_on_effective_index(df: pd.DataFrame, df_ref: pd.DataFrame):
    '''
    For classification - loss cluster
    '''
    effective_index = list(df_ref['index'].unique())
    df_new = df[df['index'].isin(effective_index)]
    return df_new


def conditional_sliced_df(df: pd.DataFrame, condition1, condition2):
    df_sliced = df[condition1 & condition2]
    return df_sliced


def insert_index_col(df):
    df.insert(0, 'index', list(df.index))
    return df


def invalid_slicing_range(slicing_input: str):
    special_characters = list(string.punctuation.replace(':', '').replace('%', ''))
    letters = list(string.ascii_letters)
    try:
        if slicing_input.split(':') == ['']:  # when user does not enter any index range info
            invalid_1 = False
        else:
            invalid_1 = ':' not in slicing_input
    except AttributeError:  # 'NoneType' object has no attribute 'split' (NoneType occurs during first spin up of this app)
        invalid_1 = True

    invalid_2 = any(char in letters for char in slicing_input)
    invalid_3 = any(char in list(slicing_input) for char in special_characters)
    if any([invalid_1, invalid_2, invalid_3]) is True:
        return True
    else:
        return False


def invalid_slicing_range_sequence(range_list: List[str]):
    start = int(range_list[0].replace('%', ''))
    stop = int(range_list[1].replace('%', ''))
    if start > stop:
        return True
    else:
        return False


def invalid_min_max_limit(range_list: List[str], df: pd.DataFrame):
    start = int(range_list[0].replace('%', ''))
    stop = int(range_list[1].replace('%', ''))

    if start < 0 or stop > len(df):
        return True
    else:
        return False


def incomplete_range_entry(range_list: List[str]):
    '''
    Either only start_idx is provided or stop_idx is provided by user
    '''
    if (any(v == '' for v in range_list)) and (range_list.count('') != 2):
        return True
    else:
        return False


def _valid_fig_key_ls(relayout_data: Dict):
    valid_key_ls = []
    for rd in relayout_data:
        try:
            if rd is None:
                valid_key_ls.append(0)
            elif ('xaxis.range[0]' in rd.keys() or 'yaxis.range[0]' in rd.keys()) and 'xaxis.autorange' not in rd.keys():
                valid_key_ls.append(1)
        except KeyError:  # related to 'xaxis.range[0]'
            valid_key_ls.append(0)
    return valid_key_ls

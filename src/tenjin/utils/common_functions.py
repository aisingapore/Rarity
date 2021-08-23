def identify_active_trace(relayout_data):
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


def is_active_trace(relayout_data):
    if ('xaxis.range[0]' in relayout_data.keys() or 'yaxis.range[0]' in relayout_data.keys()) and 'xaxis.autorange' not in relayout_data.keys():
        return True
    else:
        return False


def is_reset(relayout_data):
    if 'xaxis.autorange' in relayout_data.keys():
        return True
    else:
        return False


def detected_bimodal(models):
    if len(models) == 2:
        return True
    else:
        return False


def detected_legend_filtration(restyle_data):
    if restyle_data[0]['visible'] == ['legendonly']:
        return True
    elif restyle_data[0]['visible'] == [True]:
        return False
    else:
        return False


def detected_unique_figure(relayout_data):
    if sum(_valid_fig_key_ls(relayout_data)) == 1:
        return True
    else:
        return False


def detected_more_than_1_unique_figure(relayout_data):
    if sum(_valid_fig_key_ls(relayout_data)) > 1:
        return True
    else:
        return False


def detected_single_xaxis(relayout_data):
    if 'yaxis.range[0]' not in relayout_data.keys():
        return True
    else:
        return False


def detected_single_yaxis(relayout_data):
    if 'xaxis.range[0]' not in relayout_data.keys():
        return True
    else:
        return False


def detected_complete_pair_xaxis_yaxis(relayout_data):
    if 'xaxis.range[0]' in relayout_data.keys() and 'yaxis.range[0]' in relayout_data.keys():
        return True
    else:
        return False


def get_min_max_offset(df, models):
    min_offset = df[f'offset_{models[0]}'].min()
    max_offset = df[f'offset_{models[0]}'].max()

    if len(models) == 2:
        min_offset_m2 = df[f'offset_{models[1]}'].min()
        min_offset = min(min_offset, min_offset_m2)

        max_offset_m2 = df[f'offset_{models[1]}'].max()
        max_offset = max(max_offset, max_offset_m2)
    return min_offset, max_offset


def get_min_max_index(df, models, y_start_idx, y_stop_idx):
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


def get_min_max_cluster(df, models, y_start_idx, y_stop_idx):
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


def get_effective_xaxis_cluster(relayout_data):
    if relayout_data['xaxis.range[0]'] > 1 and relayout_data['xaxis.range[0]'] < 8:
        x_cluster = int(relayout_data['xaxis.range[0]']) + 1
    elif relayout_data['xaxis.range[0]'] <= 1:
        x_cluster = 1
    elif relayout_data['xaxis.range[0]'] >= 8:
        x_cluster = 8
    return x_cluster


def get_adjusted_xy_coordinate(relayout_data, df):
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


def dataframe_prep_on_model_count_by_yaxis_slice(df, models, y_start_idx, y_stop_idx):
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


def conditional_sliced_df(df, condition1, condition2):
    df_sliced = df[condition1 & condition2]
    return df_sliced


def insert_index_col(df):
    df.insert(0, 'index', list(df.index))
    return df


def _valid_fig_key_ls(relayout_data):
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

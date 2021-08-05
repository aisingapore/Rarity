import dash
from dash.dependencies import Input, Output, ALL
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from tenjin.app import app
from tenjin.interpreters.structured_data import IntMissPredictions
from tenjin.visualizers import miss_predictions as viz_misspred


INSTRUCTION_TEXT_SHARED = 'Click and drag on the graph to select the range of data points to inspect feature values.'
INSTRUCTION_TEXT_REG = 'To reset back to default settings, hover over icons on the top right of the graph and click "Autoscale" icon.'
WARNING_TEXT = 'To inspect new range of datapoints in different graph, please first reset the earlier selection by clicking "Autoscale" icon ' \
                'at the top right corner of the graph.'

DEFAULT_HEADER_STYLE = {'fontWeight': 'bold', 'color': 'white', 'backgroundColor': '#7e746d', 'border': '1px solid rgb(229, 211, 197)'}
DEFAULT_TITLE_STYLE = {'visibility': 'visible'}
DEFAULT_PLOT_NAME_STYLE = {'visibility': 'visible'}


def fig_plot_prediction_offset_overview(data_loader):
    df = IntMissPredictions(data_loader).xform()
    fig_obj = viz_misspred.plot_prediction_offset_overview(df)
    return fig_obj, df


def fig_probabilities_spread_pattern(data_loader):
    ls_dfs_viz, ls_class_labels, ls_dfs_by_label, ls_dfs_by_label_state = IntMissPredictions(data_loader).xform()

    # prepare fig-objs and corresponding sub-tables
    fig_objs_all_models = []
    tables_all_models = []
    for i, ls_dfs in enumerate(ls_dfs_by_label):
        fig_objs_per_model = []
        tables_per_model = []
        for j, df_specific_label in enumerate(ls_dfs):
            fig_obj = viz_misspred.plot_probabilities_spread_pattern(df_specific_label)
            fig_objs_per_model.append(fig_obj)

            table_j = viz_misspred.plot_simple_probs_spread_overview(ls_dfs_by_label_state[i][j])
            tables_per_model.append(table_j)
        fig_objs_all_models.append(fig_objs_per_model)
        tables_all_models.append(tables_per_model)

    # prepare feature table
    df_features = data_loader.get_features()
    return fig_objs_all_models, tables_all_models, ls_dfs_viz, df_features, ls_class_labels


def table_with_relayout_datapoints(data, customized_cols, header, exp_format):
    tab_obj = viz_misspred.reponsive_table_to_filtered_datapoints(data, customized_cols, header, exp_format)
    return tab_obj


def convert_relayout_data_to_df_reg(relayout_data, df, models):
    """convert raw data format from relayout selection range by user into the correct df fit for viz purpose

    Arguments:
        relayout_data {dict}: data containing selection range indices returned from plotly graph
        df {pandas dataframe}: dataframe tap-out from interpreters pipeline
        models {list}: model names defined by user during spin-up of Tenjin app

    Returns:
        pandas dataframe
        -- dataframe fit for the responsive table-graph filtering
    """
    if _detected_single_xaxis(relayout_data):
        x_start_idx = int(relayout_data['xaxis.range[0]']) if relayout_data['xaxis.range[0]'] >= 0 else 0
        x_stop_idx = int(relayout_data['xaxis.range[1]']) if relayout_data['xaxis.range[1]'] <= len(df) - 1 else len(df) - 1
        df_filtered_x = df.iloc[df.index[x_start_idx]:df.index[x_stop_idx]]

        y_start_idx, y_stop_idx = _get_min_max_offset(df_filtered_x, models)
        df_final = _dataframe_prep_on_model_count_by_yaxis_slice(df_filtered_x, models, y_start_idx, y_stop_idx)

    elif _detected_single_yaxis(relayout_data):
        y_start_idx = relayout_data['yaxis.range[0]']
        y_stop_idx = relayout_data['yaxis.range[1]']
        df_filtered_y = _dataframe_prep_on_model_count_by_yaxis_slice(df, models, y_start_idx, y_stop_idx)

        x_start_idx, x_stop_idx = _get_min_max_index(df_filtered_y, models, y_start_idx, y_stop_idx)
        x_start_idx = x_start_idx if x_start_idx >= 0 else 0
        x_stop_idx = x_stop_idx if x_stop_idx <= len(df_filtered_y) - 1 else len(df_filtered_y) - 1
        df_final = df_filtered_y.iloc[df_filtered_y.index[x_start_idx]:df_filtered_y.index[x_stop_idx]]

    else:  # a complete range is provided by user (with proper x-y coordinates)
        x_start_idx = int(relayout_data['xaxis.range[0]']) if relayout_data['xaxis.range[0]'] >= 0 else 0
        x_stop_idx = int(relayout_data['xaxis.range[1]']) if relayout_data['xaxis.range[1]'] <= len(df) - 1 else len(df) - 1
        y_start_idx = relayout_data['yaxis.range[0]']
        y_stop_idx = relayout_data['yaxis.range[1]']

        df_filtered = df.iloc[df.index[x_start_idx]:df.index[x_stop_idx]]
        df_final = _dataframe_prep_on_model_count_by_yaxis_slice(df_filtered, models, y_start_idx, y_stop_idx)
    return df_final


def convert_relayout_data_to_df_cls(fig_class_label, relayout_data, df_feature, df_viz_specific):
    """convert raw data format from relayout selection range by user into the correct df fit for viz purpose

    Arguments:
        fig_class_label {string}: class label name
        relayout_data {dict}: data containing selection range indices returned from plotly graph
        df {pandas dataframe}: dataframe tap-out from interpreters pipeline
        df_viz_specific {pandas datafrmae}: dataframe prefiltered with right class label and model

    Returns:
        pandas dataframe
        -- dataframe fit for the responsive table-graph filtering
    """
    relayout_dict = relayout_data[1]  # active relayout_data selected by user
    x_start_idx, x_stop_idx, y_start_idx, y_stop_idx = _get_adjusted_xy_coordinate(relayout_dict, df_feature)

    lower_spec_limit_x = (df_viz_specific['index'] >= x_start_idx)
    upper_spec_limit_x = (df_viz_specific['index'] <= x_stop_idx)
    df_filtered = _conditional_sliced_df(df_viz_specific, lower_spec_limit_x, upper_spec_limit_x)

    lower_spec_limit_y = (df_filtered[fig_class_label] >= y_start_idx)
    upper_spec_limit_y = (df_filtered[fig_class_label] <= y_stop_idx)
    df_final_prob = _conditional_sliced_df(df_filtered, lower_spec_limit_y, upper_spec_limit_y)

    final_filtered_idx = list(df_final_prob.index)
    df_final_feature = df_feature[df_feature['index'].isin(final_filtered_idx)]
    return df_final_feature, df_final_prob


def _insert_index_col(df):
    df.insert(0, 'index', list(df.index))
    return df


def _valid_fig_key_ls(relayout_data):
    valid_key_ls = []
    for rd in relayout_data:
        try:
            if rd is None:
                valid_key_ls.append(0)
            elif 'xaxis.range[0]' in rd.keys() or 'yaxis.range[0]' in rd.keys():
                valid_key_ls.append(1)
        except KeyError:  # related to 'xaxis.range[0]'
            valid_key_ls.append(0)
    return valid_key_ls


def _identify_active_trace(relayout_data):
    '''
    only 1 unique trace should be in relayout_data
    this function is to retrieve the position info of the unique trace
    '''
    for idx, rdata in enumerate(relayout_data):
        if rdata is not None:
            if _is_active_trace(rdata):
                specific_layout = idx, rdata
        else:
            continue
    return specific_layout


def _is_active_trace(relayout_data):
    if 'xaxis.range[0]' in relayout_data.keys() or 'yaxis.range[0]' in relayout_data.keys():
        return True
    else:
        return False


def _is_reset(relayout_data):
    if 'xaxis.autorange' in relayout_data.keys():
        return True
    else:
        return False


def _detected_single_xaxis(relayout_data):
    if 'yaxis.range[0]' not in relayout_data.keys():
        return True
    else:
        return False


def _detected_single_yaxis(relayout_data):
    if 'xaxis.range[0]' not in relayout_data.keys():
        return True
    else:
        return False


def _detected_complete_pair_xaxis_yaxis(relayout_data):
    if 'xaxis.range[0]' in relayout_data.keys() and 'yaxis.range[0]' in relayout_data.keys():
        return True
    else:
        return False


def _detected_legend_filtration(restyle_data):
    if restyle_data[0]['visible'] == ['legendonly']:
        return True
    elif restyle_data[0]['visible'] == [True]:
        return False
    else:
        return False


def _detected_unique_figure(relayout_data):
    if sum(_valid_fig_key_ls(relayout_data)) == 1:
        return True
    else:
        return False


def _detected_more_than_1_unique_figure(relayout_data):
    if sum(_valid_fig_key_ls(relayout_data)) > 1:
        return True
    else:
        return False


def _conditional_sliced_df(df, condition1, condition2):
    df_sliced = df[condition1 & condition2]
    return df_sliced


def _get_min_max_offset(df, models):
    min_offset = df[f'offset_{models[0]}'].min()
    max_offset = df[f'offset_{models[0]}'].max()

    if len(models) == 2:
        min_offset_m2 = df[f'offset_{models[1]}'].min()
        min_offset = min(min_offset, min_offset_m2)

        max_offset_m2 = df[f'offset_{models[1]}'].max()
        max_offset = max(max_offset, max_offset_m2)
    return min_offset, max_offset


def _get_min_max_index(df, models, y_start_idx, y_stop_idx):
    condition_y_min = (df[f'offset_{models[0]}'] >= y_start_idx)
    condition_y_max = (df[f'offset_{models[0]}'] <= y_stop_idx)
    df_temp = _conditional_sliced_df(df, condition_y_min, condition_y_max)
    min_index = min(list(df_temp.index))
    max_index = max(list(df_temp.index))

    if len(models) == 2:
        condition_y_min_m2 = (df[f'offset_{models[1]}'] >= y_start_idx)
        condition_y_max_m2 = (df[f'offset_{models[1]}'] <= y_stop_idx)
        df_temp_m2 = _conditional_sliced_df(df, condition_y_min_m2, condition_y_max_m2)
        min_index_m2 = min(list(df_temp_m2.index))
        min_index = min(min_index, min_index_m2)

        max_index_m2 = max(list(df_temp_m2.index))
        max_index = max(max_index, max_index_m2)
    return min_index, max_index


def _get_adjusted_xy_coordinate(relayout_data, df):
    if _detected_single_xaxis(relayout_data):
        x_start_idx = int(relayout_data['xaxis.range[0]']) if relayout_data['xaxis.range[0]'] >= 0 else 0
        x_stop_idx = int(relayout_data['xaxis.range[1]']) if relayout_data['xaxis.range[1]'] <= len(df) - 1 else len(df) - 1
        y_start_idx = 0
        y_stop_idx = 1

    elif _detected_single_yaxis(relayout_data):
        x_start_idx = 0
        x_stop_idx = len(df) - 1
        y_start_idx = relayout_data['yaxis.range[0]'] if relayout_data['yaxis.range[0]'] >= 0 else 0
        y_stop_idx = relayout_data['yaxis.range[1]'] if relayout_data['yaxis.range[1]'] <= 1 else 1

    elif _detected_complete_pair_xaxis_yaxis(relayout_data):
        x_start_idx = int(relayout_data['xaxis.range[0]']) if relayout_data['xaxis.range[0]'] >= 0 else 0
        x_stop_idx = int(relayout_data['xaxis.range[1]']) if relayout_data['xaxis.range[1]'] <= len(df) - 1 else len(df) - 1
        y_start_idx = relayout_data['yaxis.range[0]'] if relayout_data['yaxis.range[0]'] > 0 else 0
        y_stop_idx = relayout_data['yaxis.range[1]'] if relayout_data['yaxis.range[1]'] < 1 else 1
    return x_start_idx, x_stop_idx, y_start_idx, y_stop_idx


def _dataframe_prep_on_model_count_by_yaxis_slice(df, models, y_start_idx, y_stop_idx):
    offset_cols = [col for col in df.columns if 'offset_' in col]
    condition_m1_1 = (df[offset_cols[0]] >= y_start_idx)
    condition_m1_2 = (df[offset_cols[0]] <= y_stop_idx)

    if len(models) == 2:
        condition_m2_1 = (df[offset_cols[1]] >= y_start_idx)
        condition_m2_2 = (df[offset_cols[1]] <= y_stop_idx)
        df_final_m1 = _conditional_sliced_df(df, condition_m1_1, condition_m1_2)
        df_final_m2 = _conditional_sliced_df(df, condition_m2_1, condition_m2_2)

        final_filtered_idx = set(df_final_m1.index).union(set(df_final_m2.index))
        df_final = df[df['index'].isin(list(final_filtered_idx))]
    else:
        if offset_cols[0].replace('offset_', '') == models[0]:
            df_final = _conditional_sliced_df(df, condition_m1_1, condition_m1_2)
        else:
            condition_m2_1 = (df[offset_cols[1]] >= y_start_idx)
            condition_m2_2 = (df[offset_cols[1]] <= y_stop_idx)
            df_final = _conditional_sliced_df(df, condition_m2_1, condition_m2_2)
    return df_final


class MissPredictions:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.model_names = data_loader.get_model_list()
        self.is_bimodal = True if len(self.model_names) > 1 else False

        if self.analysis_type == 'regression':
            self.preds_offset, self.df = fig_plot_prediction_offset_overview(self.data_loader)
            self.cols_table_reg = [col.replace('_', ' ') for col in self.df.columns]

        elif 'classification' in self.analysis_type:
            compact_outputs = fig_probabilities_spread_pattern(self.data_loader)
            self.probs_pattern, self.label_state = compact_outputs[0], compact_outputs[1]
            self.dfs_viz, self.df_features, self.class_labels = compact_outputs[2], compact_outputs[3], compact_outputs[4]
            self.df_features = _insert_index_col(self.df_features)
            self.dfs_viz = [_insert_index_col(df) for df in self.dfs_viz]

    def show(self):
        if self.analysis_type == 'regression':
            miss_preds = dbc.Container([
                                html.Div(html.H6(INSTRUCTION_TEXT_SHARED), className='h6__dash-table-instruction-reg'),
                                dbc.Row(dcc.Graph(id='fig-reg',
                                                figure=self.preds_offset,),
                                                justify='center',
                                                className='border__common-misspred-reg'),
                                html.Div(id='alert-to-reset-reg'),
                                html.Div(id='show-feat-prob-table-reg', className='div__table-proba-misspred'),
                                html.Br()
                        ], fluid=True)

        elif 'classification' in self.analysis_type:
            fig_objs_model_1 = self.probs_pattern[0]
            tables_model_1 = self.label_state[0]

            instruction_txt_shared = [html.Div(html.H6(INSTRUCTION_TEXT_SHARED), className='h6__dash-table-instruction-cls')]
            dash_table_ls_shared = [html.Div(id='main-title-plot-name'),
                                    html.Div(id='alert-to-reset-cls'),
                                    html.Div(id='table-title-misspred-features'),
                                    html.Div(id='show-feat-table', className='div__table-proba-misspred'),
                                    html.Br(),
                                    html.Div(id='table-title-misspred-probs'),
                                    html.Div(id='show-prob-table', className='div__table-proba-misspred'),
                                    html.Br()]
            # temp_storage_shared = [dcc.Store(id='previous-active-trace')]

            if self.is_bimodal and 'classification' in self.analysis_type:  # cover bimodal_binary and bimodal_multiclass
                fig_objs_model_2 = self.probs_pattern[1]
                tables_model_2 = self.label_state[1]

                dash_fig_ls = []
                for i in range(0, len(fig_objs_model_1), 2):
                    try:  # enabling the display of a pair of figures for better comparison view
                        fig_pair = dbc.Row([
                                        dbc.Col([
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Row(
                                                        dcc.Graph(
                                                            id={'index': f'fig-cls-{self.model_names[0]}-labelcls-{self.class_labels[i]}',
                                                                'type': 'fig-obj-prob-spread'},
                                                            figure=fig_objs_model_1[i]),
                                                        justify='center'),
                                                    dbc.Row(
                                                        html.Div(
                                                            html.Div(tables_model_1[i], className='div__table-proba-misspred')),
                                                        justify='center')
                                                ]),
                                                dbc.Col([
                                                    dbc.Row(
                                                        dcc.Graph(
                                                            id={'index': f'fig-cls-{self.model_names[1]}-labelcls-{self.class_labels[i]}',
                                                                'type': 'fig-obj-prob-spread'},
                                                            figure=fig_objs_model_2[i]),
                                                        justify='center'),
                                                    dbc.Row(
                                                        html.Div(
                                                            html.Div(tables_model_2[i], className='div__table-proba-misspred')),
                                                        justify='center')
                                                ]),
                                            ])
                                        ], className='border__common'),

                                        dbc.Col([
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Row(
                                                        dcc.Graph(
                                                            id={'index': f'fig-cls-{self.model_names[0]}-labelcls-{self.class_labels[i + 1]}',
                                                                'type': 'fig-obj-prob-spread'},
                                                            figure=fig_objs_model_1[i + 1]),
                                                        justify='center'),
                                                    dbc.Row(
                                                        html.Div(
                                                            html.Div(tables_model_1[i + 1], className='div__table-proba-misspred')),
                                                        justify='center')
                                                ]),
                                                dbc.Col([
                                                    dbc.Row(
                                                        dcc.Graph(
                                                            id={'index': f'fig-cls-{self.model_names[1]}-labelcls-{self.class_labels[i + 1]}',
                                                                'type': 'fig-obj-prob-spread'},
                                                            figure=fig_objs_model_2[i + 1]),
                                                        justify='center'),
                                                    dbc.Row(
                                                        html.Div(
                                                            html.Div(tables_model_2[i + 1], className='div__table-proba-misspred')),
                                                        justify='center')
                                                ]),
                                            ])
                                        ], className='border__common')
                                    ])
                        dash_fig_ls.append(fig_pair)

                    except IndexError:  # handling the last odd figure that can't be paired out
                        fig_pair = dbc.Row([
                                        dbc.Col([
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Row(
                                                        dcc.Graph(
                                                            id={'index': f'fig-cls-{self.model_names[0]}-labelcls-{self.class_labels[i]}',
                                                                'type': 'fig-obj-prob-spread'},
                                                            figure=fig_objs_model_1[i]),
                                                        justify='center'),
                                                    dbc.Row(
                                                        html.Div(
                                                            html.Div(tables_model_1[i], className='div__table-proba-misspred')),
                                                        justify='center')
                                                ]),
                                                dbc.Col([
                                                    dbc.Row(
                                                        dcc.Graph(
                                                            id={'index': f'fig-cls-{self.model_names[1]}-labelcls-{self.class_labels[i]}',
                                                                'type': 'fig-obj-prob-spread'},
                                                            figure=fig_objs_model_2[i]),
                                                        justify='center'),
                                                    dbc.Row(
                                                        html.Div(
                                                            html.Div(tables_model_2[i], className='div__table-proba-misspred')),
                                                        justify='center')
                                                ])
                                            ])
                                        ], className='border__common')])
                        dash_fig_ls.append(fig_pair)

                compiled_fig_table_objs = instruction_txt_shared + dash_fig_ls + dash_table_ls_shared
                miss_preds = dbc.Container(compiled_fig_table_objs, fluid=True)

            elif not self.is_bimodal and 'binary' in self.analysis_type:  # single modal binary classification
                dash_fig_ls = dbc.Row([
                                    dbc.Col([
                                        dbc.Row(
                                            dcc.Graph(
                                                id={'index': f'fig-cls-{self.model_names[0]}-labelcls-{self.class_labels[0]}',
                                                    'type': 'fig-obj-prob-spread'},
                                                figure=fig_objs_model_1[0]),
                                            justify='center'),
                                        dbc.Row(html.Div(
                                            html.Div(tables_model_1[0], className='div__table-proba-misspred')), justify='center'),
                                    ]),
                                    dbc.Col([
                                        dbc.Row(
                                            dcc.Graph(
                                                id={'index': f'fig-cls-{self.model_names[0]}-labelcls-{self.class_labels[1]}',
                                                    'type': 'fig-obj-prob-spread'},
                                                figure=fig_objs_model_1[1]),
                                            justify='center'),
                                        dbc.Row(
                                            html.Div(
                                                html.Div(tables_model_1[1], className='div__table-proba-misspred')),
                                            justify='center'),
                                    ])], className='border__common-misspred-cls-single-binary')

                instruction_txt_cls_single_binary = [html.Div(
                                                        html.H6(INSTRUCTION_TEXT_SHARED),
                                                        className='h6__dash-table-instruction-cls-single-binary')]

                compiled_fig_table_objs = instruction_txt_cls_single_binary + [dash_fig_ls] + dash_table_ls_shared
                miss_preds = dbc.Container(compiled_fig_table_objs, fluid=True)

            elif not self.is_bimodal and 'multiclass' in self.analysis_type:  # single modal multi-class classification
                dash_fig_ls = []
                for i in range(0, len(fig_objs_model_1)):
                    fig_pair = dbc.Col([
                                    dbc.Row([dcc.Graph(id={'index': f'fig-cls-{self.model_names[0]}-labelcls-{self.class_labels[i]}',
                                                        'type': 'fig-obj-prob-spread'},
                                                    figure=fig_objs_model_1[i])], justify='center'),
                                    dbc.Row(
                                        html.Div(
                                            html.Div(tables_model_1[i], className='div__table-proba-misspred')),
                                        justify='center'),
                                ], className='border__common')
                    dash_fig_ls.append(fig_pair)

                compiled_fig_table_objs = instruction_txt_shared + [dbc.Row(dash_fig_ls)] + dash_table_ls_shared
                miss_preds = dbc.Container(compiled_fig_table_objs, fluid=True)
        return miss_preds

    # callback for regression tasks
    def callback(self):
        @app.callback(
            Output('alert-to-reset-reg', 'children'),
            Output('show-feat-prob-table-reg', 'children'),
            Input('fig-reg', 'relayoutData'),  # taking in range selection info from figures
            Input('fig-reg', 'restyleData'))  # taking in legend filtration info from figures
        def display_relayout_data_reg(relayout_data, restyle_data):
            if relayout_data is not None:
                try:
                    self.df = self.df.round(2)  # to limit table cell having values with long decimals for better viz purpose
                except TypeError:
                    self.df

                if _is_reset(relayout_data):
                    DEFAULT_HEADER_STYLE['border'] = 'none'
                    DEFAULT_HEADER_STYLE['visibility'] = 'collapse'
                    table_obj_reg = table_with_relayout_datapoints('', self.cols_table_reg, DEFAULT_HEADER_STYLE, 'none')
                    alert_obj_reg = dbc.Alert(color="light", style={'visibility': 'hidden'})

                elif _is_active_trace(relayout_data):
                    models = self.model_names
                    if restyle_data is not None:  # [{'visible': ['legendonly']}, [1]]
                        if _detected_legend_filtration(restyle_data):
                            model_to_exclude_from_view = self.model_names[restyle_data[1][0]]
                            models = [model for model in self.model_names if model != model_to_exclude_from_view]

                    df_final = convert_relayout_data_to_df_reg(relayout_data, self.df, models)
                    df_final.columns = self.cols_table_reg  # to have customized column names displayed on table

                    DEFAULT_HEADER_STYLE['visibility'] = 'visible'
                    data_relayout_reg = df_final.to_dict('records')
                    table_obj_reg = table_with_relayout_datapoints(data_relayout_reg, self.cols_table_reg, DEFAULT_HEADER_STYLE, 'csv')
                    alert_obj_reg = dbc.Alert(INSTRUCTION_TEXT_REG, color="secondary", className='alert__note-reg')
                return alert_obj_reg, table_obj_reg
            else:
                raise PreventUpdate

        # callback for classification tasks
        @app.callback(
            Output('alert-to-reset-cls', 'children'),
            Output('main-title-plot-name', 'children'),
            Output('table-title-misspred-features', 'children'),
            Output('show-feat-table', 'children'),
            Output('table-title-misspred-probs', 'children'),
            Output('show-prob-table', 'children'),
            Input({"index": ALL, "type": "fig-obj-prob-spread"}, 'relayoutData'),
            Input({"index": ALL, "type": "fig-obj-prob-spread"}, 'restyleData'))
        def display_relayout_data_cls(relayout_data, restyle_data):
            fig_obj_ids = [fig_id_dict['id']['index'] for fig_id_dict in dash.callback_context.inputs_list[0]]

            title_main_plot_name = html.H5('', style=DEFAULT_PLOT_NAME_STYLE, className='title__main-plot-name-cls')
            title_table_features = html.H6('Feature Values :', style=DEFAULT_TITLE_STYLE, className='title__table-misspred-cls')
            title_table_probs = html.H6('Probabilities Overview :', style=DEFAULT_TITLE_STYLE, className='title__table-misspred-cls')

            if not all(item is None for item in relayout_data):
                try:
                    self.df_features = self.df_features.round(2)  # limit long decimals on feature values
                    self.dfs_viz[0] = self.dfs_viz[0].round(4)  # standardize prob values to 4 decimals
                    if self.is_bimodal:
                        self.dfs_viz[1] = self.dfs_viz[1].round(4)
                except TypeError:
                    self.df_features
                    self.dfs_viz

                models_ref_dict = {model: i for i, model in enumerate(self.model_names)}

                # default to invisible for reset / Autoscale scenario
                DEFAULT_HEADER_STYLE['border'] = 'none'
                DEFAULT_HEADER_STYLE['visibility'] = 'collapse'
                DEFAULT_TITLE_STYLE['visibility'] = 'hidden'
                DEFAULT_PLOT_NAME_STYLE['visibility'] = 'hidden'

                table_obj_cls_features = table_with_relayout_datapoints('', list(self.df_features.columns), DEFAULT_HEADER_STYLE, 'none')
                table_obj_cls_probs = table_with_relayout_datapoints('', list(self.dfs_viz[0].columns), DEFAULT_HEADER_STYLE, 'none')

                if _detected_unique_figure(relayout_data):  # unique fig trace (only one fig is used to generate info table)
                    specific_relayout = _identify_active_trace(relayout_data)  # output as tuple (idx, relayout_data)

                    fig_id = fig_obj_ids[specific_relayout[0]]
                    fig_class_label = fig_id.split('-cls-')[-1].split('-labelcls-')[-1]
                    model_in_view = fig_id.split('-cls-')[-1].split('-labelcls-')[0]
                    title_main_plot_name = html.H5(f'Currently inspecting : class {fig_class_label} [ {model_in_view} ]',
                                                    style=DEFAULT_PLOT_NAME_STYLE,
                                                    className='title__main-plot-name-cls')

                    specific_df_viz_id = models_ref_dict[model_in_view]
                    df_viz_model_in_view = self.dfs_viz[specific_df_viz_id]
                    df_viz_specific = df_viz_model_in_view[df_viz_model_in_view['yTrue'].astype('str') == fig_class_label]

                    specific_restyle_data = restyle_data[specific_relayout[0]]
                    if specific_restyle_data is not None:
                        if _detected_legend_filtration(specific_restyle_data):
                            if specific_restyle_data[1][0] == 1:
                                data_field_to_exclude = 'miss-predict'
                            else:
                                data_field_to_exclude = 'correct'
                            df_viz_specific = df_viz_specific[df_viz_specific['pred_state'] != data_field_to_exclude]
                        else:
                            df_viz_specific

                    df_filtered_feature, df_filtered_viz = convert_relayout_data_to_df_cls(fig_class_label,
                                                                                            specific_relayout,
                                                                                            self.df_features,
                                                                                            df_viz_specific)

                    df_filtered_viz.columns = [col.replace('_', ' ') for col in df_filtered_viz.columns]
                    data_relayout_features = df_filtered_feature.to_dict('records')

                    # activate visibility to prepare rendering of data table upon completion of range selection and/or legend filtration
                    DEFAULT_HEADER_STYLE['visibility'] = 'visible'
                    DEFAULT_TITLE_STYLE['visibility'] = 'visible'
                    DEFAULT_PLOT_NAME_STYLE['visibility'] = 'visible'

                    title_main_plot_name = html.H5(f'Currently inspecting : class {fig_class_label} [ {model_in_view} ]',
                                                    style=DEFAULT_PLOT_NAME_STYLE,
                                                    className='title__main-plot-name-cls')
                    table_obj_cls_features = table_with_relayout_datapoints(data_relayout_features,
                                                                            list(self.df_features.columns),
                                                                            DEFAULT_HEADER_STYLE,
                                                                            'csv')

                    data_relayout_probs = df_filtered_viz.to_dict('records')
                    table_obj_cls_probs = table_with_relayout_datapoints(data_relayout_probs,
                                                                        list(df_filtered_viz.columns),
                                                                        DEFAULT_HEADER_STYLE,
                                                                        'csv')
                    alert_obj_cls = dbc.Alert(color="light")
                elif _detected_more_than_1_unique_figure(relayout_data):
                    alert_obj_cls = dbc.Alert(WARNING_TEXT, color="danger", style={'text-align': 'center'})
                else:  # user reset data range
                    alert_obj_cls = dbc.Alert(color="light")

                return alert_obj_cls, title_main_plot_name, title_table_features, table_obj_cls_features, title_table_probs, table_obj_cls_probs

            else:
                raise PreventUpdate

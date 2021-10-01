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

from typing import Union

import dash
from dash.dependencies import Input, Output, ALL
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from rarity.app import app
from rarity.data_loader import CSVDataLoader, DataframeLoader
from rarity.interpreters.structured_data import IntMissPredictions
from rarity.visualizers import miss_predictions as viz_misspred
from rarity.visualizers import shared_viz_component as viz_shared
from rarity.utils import style_configs
from rarity.utils.common_functions import (identify_active_trace, is_active_trace, is_reset, is_regression, is_classification,
                                            detected_legend_filtration, detected_unique_figure, detected_more_than_1_unique_figure,
                                            detected_single_xaxis, detected_single_yaxis, get_min_max_index, get_min_max_offset,
                                            get_adjusted_xy_coordinate, conditional_sliced_df, dataframe_prep_on_model_count_by_yaxis_slice,
                                            insert_index_col)


def fig_plot_prediction_offset_overview(data_loader: Union[CSVDataLoader, DataframeLoader]):
    '''
    For use in regression task only.
    Display scatter plot for overview on prediction offset values

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Returns:
        :obj:`~plotly.graph_objects.Figure`:
            figure displaying scatter plot outlining overview on prediction offset values by index
    '''
    df = IntMissPredictions(data_loader).xform()
    fig_obj = viz_misspred.plot_prediction_offset_overview(df)
    return fig_obj, df


def fig_probabilities_spread_pattern(data_loader: Union[CSVDataLoader, DataframeLoader]):
    '''
    For use in classification task only.
    Function to output collated info packs used to display final graph objects and data tables

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Returns:

            Compact outputs consist of the followings

            - fig_objs_all_models (:obj: `List[~plotly.graph_objects.Figure]`): figure displaying scatter plot outlining probabilities \
                comparison on correct data point vs miss-predicted data point for each class label
            - tables_all_models (:obj:`List[~dash_table.DataTable]`): table object outlining simple stats on ss, %correct, % wrong, accuracy \
                for each label class
            - ls_dfs_viz (:obj:`List[~pandas.DataFrame]`): dataframes for overview visualization need with true labels and \
                predicted labels included
            - df_features (:obj:`~pandas.DataFrame`): dataframe storing all features used in dataset
            - ls_class_labels (:obj:`List[str]`): list of class labels found in dataset
    '''
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
    '''
    Create table outlining dataframe content

    Arguments:
        data (:obj:`~dash_table.DataTable`):
            dictionary like format storing dataframe info under 'record' key
        customized_cols (:obj:`List[str]`):
            list of customized column names
        header (:obj:`Dict`):
            dictionary format storing the style info for table header
        exp_format (str):
            text info indicating the export format

    Returns:
        :obj:`~dash_table.DataTable`:
            table object outlining the dataframe content with specific styles
    '''
    tab_obj = viz_shared.reponsive_table_to_filtered_datapoints(data, customized_cols, header, exp_format)
    return tab_obj


def convert_relayout_data_to_df_reg(relayout_data, df, models):
    '''
    Convert raw data format from relayout selection range by user into the correct df fit for viz purpose

    Arguments:
        relayout_data (:obj:`Dict`):
            dictionary like data containing selection range indices returned from plotly graph
        df (:obj:`~pandas.DataFrame`):
            dataframe tap-out from interpreters pipeline
        models (:obj:`List[str]`):
            model names defined by user during spin-up of Tenjin app

    Returns:
        :obj:`~pandas.DataFrame`:
            dataframe fit for the responsive table-graph filtering
    '''
    if detected_single_xaxis(relayout_data):
        x_start_idx = int(relayout_data['xaxis.range[0]']) if relayout_data['xaxis.range[0]'] >= 0 else 0
        x_stop_idx = int(relayout_data['xaxis.range[1]']) if relayout_data['xaxis.range[1]'] <= len(df) - 1 else len(df) - 1
        df_filtered_x = df.iloc[df.index[x_start_idx]:df.index[x_stop_idx]]

        y_start_idx, y_stop_idx = get_min_max_offset(df_filtered_x, models)
        df_final = dataframe_prep_on_model_count_by_yaxis_slice(df_filtered_x, models, y_start_idx, y_stop_idx)

    elif detected_single_yaxis(relayout_data):
        y_start_idx = relayout_data['yaxis.range[0]']
        y_stop_idx = relayout_data['yaxis.range[1]']
        df_filtered_y = dataframe_prep_on_model_count_by_yaxis_slice(df, models, y_start_idx, y_stop_idx)

        x_start_idx, x_stop_idx = get_min_max_index(df_filtered_y, models, y_start_idx, y_stop_idx)
        x_start_idx = x_start_idx if x_start_idx >= 0 else 0
        x_stop_idx = x_stop_idx if x_stop_idx <= len(df_filtered_y) - 1 else len(df_filtered_y) - 1
        df_final = df_filtered_y.iloc[df_filtered_y.index[x_start_idx]:df_filtered_y.index[x_stop_idx]]

    else:  # a complete range is provided by user (with proper x-y coordinates)
        x_start_idx = int(relayout_data['xaxis.range[0]']) if relayout_data['xaxis.range[0]'] >= 0 else 0
        x_stop_idx = int(relayout_data['xaxis.range[1]']) if relayout_data['xaxis.range[1]'] <= len(df) - 1 else len(df) - 1
        y_start_idx = relayout_data['yaxis.range[0]']
        y_stop_idx = relayout_data['yaxis.range[1]']

        df_filtered = df.iloc[df.index[x_start_idx]:df.index[x_stop_idx]]
        df_final = dataframe_prep_on_model_count_by_yaxis_slice(df_filtered, models, y_start_idx, y_stop_idx)
    return df_final


def convert_relayout_data_to_df_cls(fig_class_label, relayout_data, df_feature, df_viz_specific):
    '''
    Convert raw data format from relayout selection range by user into the correct df fit for viz purpose

    Arguments:
        fig_class_label (str):
            class label name
        relayout_data (:obj:`Dict`):
            data containing selection range indices returned from plotly graph
        df (:obj:`~pandas.DataFrame`):
            dataframe tap-out from interpreters pipeline
        df_viz_specific (:obj:`~pandas.DataFrame`):
            dataframe prefiltered with right class label and model

    Returns:
        :obj:`~pandas.DataFrame`:
            dataframe fit for the responsive table-graph filtering
    '''
    relayout_dict = relayout_data[1]  # active relayout_data selected by user
    x_start_idx, x_stop_idx, y_start_idx, y_stop_idx = get_adjusted_xy_coordinate(relayout_dict, df_feature)

    lower_spec_limit_x = (df_viz_specific['index'] >= x_start_idx)
    upper_spec_limit_x = (df_viz_specific['index'] <= x_stop_idx)
    df_filtered = conditional_sliced_df(df_viz_specific, lower_spec_limit_x, upper_spec_limit_x)

    lower_spec_limit_y = (df_filtered[fig_class_label] >= y_start_idx)
    upper_spec_limit_y = (df_filtered[fig_class_label] <= y_stop_idx)
    df_final_prob = conditional_sliced_df(df_filtered, lower_spec_limit_y, upper_spec_limit_y)

    final_filtered_idx = list(df_final_prob.index)
    df_final_feature = df_feature[df_feature['index'].isin(final_filtered_idx)]
    return df_final_feature, df_final_prob


class MissPredictions:
    '''
    Main integration for feature component on Miss Prediction.

        - On Regression: To generate single miss-prediction scatter plot by data index points
        - On Classification: To generate scatter plots for probabilities comparison on correct data point vs miss-predicted data point \
            for each class label

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Important Attributes:

        - analysis_type (str):
            Analysis type defined by user during initial inputs preparation via data_loader stage.
        - model_names (:obj:`List[str]`):
            model names defined by user during initial inputs preparation via data_loader stage.
        - is_bimodal (bool):
            to indicate if analysis involves 2 models

    Returns:
        :obj:`~dash_core_components.Container`:
            styled dash components displaying graph and/or table objects
    '''
    def __init__(self, data_loader: Union[CSVDataLoader, DataframeLoader]):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.model_names = data_loader.get_model_list()
        self.is_bimodal = True if len(self.model_names) > 1 else False

        # instantiate here instead of under def show() as it will be used in callbacks as well
        if is_regression(self.analysis_type):
            self.preds_offset, self.df = fig_plot_prediction_offset_overview(self.data_loader)
            self.cols_table_reg = [col.replace('_', ' ') for col in self.df.columns]

        elif is_classification(self.analysis_type):
            compact_outputs = fig_probabilities_spread_pattern(self.data_loader)
            self.probs_pattern, self.label_state = compact_outputs[0], compact_outputs[1]
            self.dfs_viz, self.df_features, self.class_labels = compact_outputs[2], compact_outputs[3], compact_outputs[4]
            self.df_features = insert_index_col(self.df_features)
            self.dfs_viz = [insert_index_col(df) for df in self.dfs_viz]

    def show(self):
        '''
        Method to tapout styled html for misspredictions
        '''
        if is_regression(self.analysis_type):
            miss_preds = dbc.Container([
                                html.Div(html.H6(style_configs.INSTRUCTION_TEXT_SHARED), className='h6__dash-table-instruction-reg'),
                                dbc.Row(dcc.Graph(id='fig-reg',
                                                figure=self.preds_offset,),
                                                justify='center',
                                                className='border__common-misspred-reg'),
                                html.Div(id='alert-to-reset-reg'),
                                html.Div(id='show-feat-prob-table-reg', className='div__table-proba-misspred'),
                                html.Br()
                        ], fluid=True)

        elif is_classification(self.analysis_type):
            fig_objs_model_1 = self.probs_pattern[0]
            tables_model_1 = self.label_state[0]

            instruction_txt_shared = [html.Div(html.H6(style_configs.INSTRUCTION_TEXT_SHARED),
                                                        className='h6__dash-table-instruction-misspred-cls')]
            dash_table_ls_shared = [html.Div(id='main-title-plot-name'),
                                    html.Div(id='alert-to-reset-cls'),
                                    html.Div(id='table-title-misspred-features'),
                                    html.Div(id='show-feat-table', className='div__table-proba-misspred'),
                                    html.Br(),
                                    html.Div(id='table-title-misspred-probs'),
                                    html.Div(id='show-prob-table', className='div__table-proba-misspred'),
                                    html.Br()]

            if self.is_bimodal and is_classification(self.analysis_type):  # cover bimodal_binary and bimodal_multiclass
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
                                        ], className='border__common-left'),

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
                                        ], className='border__common-right')
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
                                                        html.H6(style_configs.INSTRUCTION_TEXT_SHARED),
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

    def callbacks(self):
        @app.callback(
            Output('alert-to-reset-reg', 'children'),
            Output('show-feat-prob-table-reg', 'children'),
            Input('fig-reg', 'relayoutData'),  # taking in range selection info from figures
            Input('fig-reg', 'restyleData'))  # taking in legend filtration info from figures
        def display_relayout_data_reg(relayout_data, restyle_data):
            '''
            Callbacks functionalities for regression tasks
            '''
            if relayout_data is not None:
                try:
                    self.df = self.df.round(2)  # to limit table cell having values with long decimals for better viz purpose
                except TypeError:
                    self.df

                if is_reset(relayout_data):
                    collapsed_header = style_configs.collapse_header_style()

                    table_obj_reg = table_with_relayout_datapoints('', self.cols_table_reg, collapsed_header, 'none')
                    alert_obj_reg = style_configs.dummy_alert()

                elif is_active_trace(relayout_data):
                    models = self.model_names
                    if restyle_data is not None:  # [{'visible': ['legendonly']}, [1]]
                        if detected_legend_filtration(restyle_data):
                            model_to_exclude_from_view = self.model_names[restyle_data[1][0]]
                            models = [model for model in self.model_names if model != model_to_exclude_from_view]

                    df_final = convert_relayout_data_to_df_reg(relayout_data, self.df, models)
                    df_final.columns = self.cols_table_reg  # to have customized column names displayed on table

                    default_header = style_configs.default_header_style()
                    data_relayout_reg = df_final.to_dict('records')
                    table_obj_reg = table_with_relayout_datapoints(data_relayout_reg, self.cols_table_reg, default_header, 'csv')
                    alert_obj_reg = style_configs.activate_alert()
                return alert_obj_reg, table_obj_reg
            else:
                raise PreventUpdate

        @app.callback(
            Output('alert-to-reset-cls', 'children'),
            Output('main-title-plot-name', 'children'),
            Output('table-title-misspred-features', 'children'),
            Output('show-feat-table', 'children'),
            Output('table-title-misspred-probs', 'children'),
            Output('show-prob-table', 'children'),
            Input({'index': ALL, 'type': 'fig-obj-prob-spread'}, 'relayoutData'),
            Input({'index': ALL, 'type': 'fig-obj-prob-spread'}, 'restyleData'))
        def display_relayout_data_cls(relayout_data, restyle_data):
            '''
            Callbacks functionalities for classification tasks
            '''
            fig_obj_ids = [fig_id_dict['id']['index'] for fig_id_dict in dash.callback_context.inputs_list[0]]

            default_plot_name = style_configs.DEFAULT_PLOT_NAME_STYLE
            default_title = style_configs.DEFAULT_TITLE_STYLE
            title_main_plot_name = html.H5('', style=default_plot_name, className='title__main-plot-name-cls')
            title_table_features = html.H6('Feature Values :', style=default_title, className='title__table-misspred-cls')
            title_table_probs = html.H6('Probabilities Overview :', style=default_title, className='title__table-misspred-cls')

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

                collapsed_header = style_configs.collapse_header_style()
                default_title = style_configs.hidden_title_style()
                default_plot_name = style_configs.hidden_plot_name_style()

                table_obj_cls_features = table_with_relayout_datapoints('', list(self.df_features.columns), collapsed_header, 'none')
                table_obj_cls_probs = table_with_relayout_datapoints('', list(self.dfs_viz[0].columns), collapsed_header, 'none')

                if detected_unique_figure(relayout_data):  # unique fig trace (only one fig is used to generate info table)
                    specific_relayout = identify_active_trace(relayout_data)  # output as tuple (idx, relayout_data)

                    fig_id = fig_obj_ids[specific_relayout[0]]
                    fig_class_label = fig_id.split('-cls-')[-1].split('-labelcls-')[-1]
                    model_in_view = fig_id.split('-cls-')[-1].split('-labelcls-')[0]
                    title_main_plot_name = html.H5(f'Currently inspecting : class {fig_class_label} [ {model_in_view} ]',
                                                    style=default_plot_name,
                                                    className='title__main-plot-name-cls')

                    specific_df_viz_id = models_ref_dict[model_in_view]
                    df_viz_model_in_view = self.dfs_viz[specific_df_viz_id]
                    df_viz_specific = df_viz_model_in_view[df_viz_model_in_view['yTrue'].astype('str') == fig_class_label]

                    specific_restyle_data = restyle_data[specific_relayout[0]]
                    if specific_restyle_data is not None:
                        if detected_legend_filtration(specific_restyle_data):
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
                    default_header = style_configs.default_header_style()
                    default_title['visibility'] = 'visible'
                    default_plot_name['visibility'] = 'visible'

                    title_main_plot_name = html.H5(f'Currently inspecting : class {fig_class_label} [ {model_in_view} ]',
                                                    style=default_plot_name,
                                                    className='title__main-plot-name-cls')
                    table_obj_cls_features = table_with_relayout_datapoints(data_relayout_features,
                                                                            list(self.df_features.columns),
                                                                            default_header,
                                                                            'csv')

                    data_relayout_probs = df_filtered_viz.to_dict('records')
                    table_obj_cls_probs = table_with_relayout_datapoints(data_relayout_probs,
                                                                        list(df_filtered_viz.columns),
                                                                        default_header,
                                                                        'csv')
                    alert_obj_cls = dbc.Alert(color="light")
                elif detected_more_than_1_unique_figure(relayout_data):
                    alert_obj_cls = dbc.Alert(style_configs.WARNING_TEXT, color="danger", style={'text-align': 'center'})
                else:  # user reset data range
                    alert_obj_cls = dbc.Alert(color="light")

                return alert_obj_cls, title_main_plot_name, title_table_features, table_obj_cls_features, title_table_probs, table_obj_cls_probs

            else:
                raise PreventUpdate

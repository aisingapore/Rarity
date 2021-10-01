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

from typing import Union, List, Dict
import math
import pandas as pd

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from rarity.app import app
from rarity.data_loader import CSVDataLoader, DataframeLoader
from rarity.interpreters.structured_data import IntLossClusterer
from rarity.visualizers import loss_clusters as viz_clusters
from rarity.visualizers import shared_viz_component as viz_shared
from rarity.utils import style_configs
from rarity.utils.common_functions import (is_active_trace, is_reset, is_regression, is_classification,
                                            detected_legend_filtration, detected_single_xaxis, detected_single_yaxis,
                                            detected_bimodal, get_min_max_offset, get_min_max_cluster, get_effective_xaxis_cluster,
                                            get_adjusted_dfs_based_on_legend_filtration, conditional_sliced_df, insert_index_col,
                                            dataframe_prep_on_model_count_by_yaxis_slice, new_dataframe_prep_based_on_effective_index)


def fig_plot_offset_clusters_reg(data_loader: Union[CSVDataLoader, DataframeLoader], num_cluster: int):
    '''
    For use in regression task only.
    Function to output collated info packs used to display final graph objects by cluster groups along with calculated silhouette scores

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module
        num_cluster (int):
            Number of cluster to form

    Returns:

             Compact outputs consist of the followings

            - df (:obj:`~pd.DataFrame`): dataframes for overview visualization need with offset values included
            - fig_obj_cluster (:obj:`~plotly.graph_objects.Figure`): figure displaying violin plot outlining cluster groups by offset values
            - ls_cluster_score (:obj:`List[str]`): list of silhouette scores, indication of clustering quality
            - fig_obj_elbow (:obj:`~plotly.graph_objects.Figure`): figure displaying line plot outlining the change in sum of squared distances \
                along the cluster range
    '''
    df, ls_cluster_score, ls_cluster_range, ls_ssd = IntLossClusterer(data_loader).xform(num_cluster, None, 'All')
    models = data_loader.get_model_list()
    analysis_type = data_loader.get_analysis_type()

    fig_obj_cluster = viz_clusters.plot_offset_clusters(df, analysis_type)
    fig_obj_elbow = viz_clusters.plot_optimum_cluster_via_elbow_method(ls_cluster_range, ls_ssd, models)
    return df, fig_obj_cluster, ls_cluster_score, fig_obj_elbow


def fig_plot_logloss_clusters_cls(data_loader: Union[CSVDataLoader, DataframeLoader],
                                    num_cluster: int,
                                    log_func: math.log = math.log,
                                    specific_dataset: str = 'All'):
    '''
    For use in classification task only.
    Function to output collated info packs used to display final graph objects by cluster groups along with calculated silhouette scores

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module
        num_cluster (int):
            Number of cluster to form
        log_funct (:obj:`math.log`):
            Mathematics logarithm function used to calculate log-loss between yTrue and yPred
        specific_dataset (str):
            Default to 'All' indicating to include all miss-predict labels. Other options flexibly expand depending on class labels

    Returns:

             Compact outputs consist of the followings

            - ls_dfs_viz (:obj:`List[~pd.DataFrame]`): dataframes for overview visualization need with offset values included
            - fig_obj_cluster (:obj:`~plotly.graph_objects.Figure`): figure displaying violin plot outlining cluster groups by offset values
            - ls_cluster_score (:obj:`List[str]`): list of silhouette scores, indication of clustering quality
            - fig_obj_elbow (:obj:`~plotly.graph_objects.Figure`): figure displaying line plot outlining the change in sum of squared distances \
                along the cluster range
            - ls_class_labels (:obj:`List[str]`): list of all class labels
            - ls_class_labels_misspred (:obj:`List[str]`): list of class labels with minimum of 1 miss-prediction
            - df_features (:obj:`~pandas.DataFrame`): dataframe storing all features used in dataset
    '''
    compact_outputs = IntLossClusterer(data_loader).xform(num_cluster, log_func, specific_dataset)
    ls_dfs_viz, ls_class_labels, ls_class_labels_misspred = compact_outputs[0], compact_outputs[1], compact_outputs[2]
    ls_cluster_score, ls_cluster_range, ls_ssd = compact_outputs[3], compact_outputs[4], compact_outputs[5]
    df_features = data_loader.get_features()
    analysis_type = data_loader.get_analysis_type()
    models = data_loader.get_model_list()

    fig_obj_cluster = viz_clusters.plot_logloss_clusters(ls_dfs_viz, analysis_type)
    fig_obj_elbow = viz_clusters.plot_optimum_cluster_via_elbow_method(ls_cluster_range, ls_ssd, models)
    return ls_dfs_viz, fig_obj_cluster, ls_cluster_score, fig_obj_elbow, ls_class_labels, ls_class_labels_misspred, df_features


def table_with_relayout_datapoints(data: dash_table.DataTable, customized_cols: List[str], header: Dict, exp_format: str):
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


def convert_cluster_relayout_data_to_df_reg(relayout_data: Dict, df: pd.DataFrame, models: List[str]):
    '''
    For use in regression task only.
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
        x_cluster = get_effective_xaxis_cluster(relayout_data)
        df_filtered_x = df[df[f'cluster_{models[0]}'] == x_cluster]

        if detected_bimodal(models):
            df_filtered_x_m2 = df[df[f'cluster_{models[1]}'] == x_cluster]
            df_filtered_x = pd.concat([df_filtered_x, df_filtered_x_m2]).drop_duplicates()

        y_start_idx, y_stop_idx = get_min_max_offset(df_filtered_x, models)
        df_final = dataframe_prep_on_model_count_by_yaxis_slice(df_filtered_x, models, y_start_idx, y_stop_idx)

    elif detected_single_yaxis(relayout_data):
        y_start_idx = relayout_data['yaxis.range[0]']
        y_stop_idx = relayout_data['yaxis.range[1]']
        df_filtered_y = dataframe_prep_on_model_count_by_yaxis_slice(df, models, y_start_idx, y_stop_idx)

        x_start_idx, x_stop_idx = get_min_max_cluster(df_filtered_y, models, y_start_idx, y_stop_idx)
        x_start_idx = x_start_idx if x_start_idx >= 1 else 1
        x_stop_idx = x_stop_idx if x_stop_idx <= 8 else 8

        condition_min_cluster = df_filtered_y[f'cluster_{models[0]}'] >= x_start_idx
        condition_max_cluster = df_filtered_y[f'cluster_{models[0]}'] <= x_stop_idx
        df_final = conditional_sliced_df(df_filtered_y, condition_min_cluster, condition_max_cluster)

        if detected_bimodal(models):
            condition_min_cluster_m2 = df_filtered_y[f'cluster_{models[1]}'] >= x_start_idx
            condition_max_cluster_m2 = df_filtered_y[f'cluster_{models[1]}'] <= x_stop_idx
            df_final_m2 = conditional_sliced_df(df_filtered_y, condition_min_cluster_m2, condition_max_cluster_m2)
            df_final = pd.concat([df_final, df_final_m2]).drop_duplicates()

    else:  # a complete range is provided by user (with proper x-y coordinates)
        x_cluster = get_effective_xaxis_cluster(relayout_data)
        y_start_idx = relayout_data['yaxis.range[0]']
        y_stop_idx = relayout_data['yaxis.range[1]']
        df_filtered = df[df[f'cluster_{models[0]}'] == x_cluster]

        if detected_bimodal(models):
            df_filtered_m2 = df[df[f'cluster_{models[1]}'] == x_cluster]
            df_filtered = pd.concat([df_filtered, df_filtered_m2]).drop_duplicates()

        df_final = dataframe_prep_on_model_count_by_yaxis_slice(df_filtered, models, y_start_idx, y_stop_idx)
    return df_final


def convert_cluster_relayout_data_to_df_cls(relayout_data: Dict, dfs_viz: List[pd.DataFrame], df_features: pd.DataFrame, models: List[str]):
    '''
    For use in classification task only.
    Convert raw data format from relayout selection range by user into the correct df fit for viz purpose

    Arguments:
        relayout_data (:obj:`Dict`):
            dictionary like data containing selection range indices returned from plotly graph
        dfs_viz (:obj:`List[~pd.DataFrame]`):
            list of dataframes for overview visualization need with offset values included
        df_features (:obj:`~pandas.DataFrame`):
            dataframe storing all features used in dataset
        models (:obj:`List[str]`):
            model names defined by user during spin-up of Tenjin app

    Returns:

             Compact outputs consist of the followings

            - df_final_features (:obj:`~pd.DataFrame`): dataframe storing all features based on slicing info from relayout_data
            - df_final_probs (:obj:`~pd.DataFrame`): dataframe storing probability values by class label corresponding to \
                the slicing relayout_data
    '''
    if detected_single_xaxis(relayout_data):
        x_cluster = get_effective_xaxis_cluster(relayout_data)
        df_final_probs = dfs_viz[0][dfs_viz[0]['cluster'] == x_cluster]

        if detected_bimodal(models):
            df_final_probs_m2 = dfs_viz[1][dfs_viz[1]['cluster'] == x_cluster]
            df_final_probs = pd.concat([df_final_probs, df_final_probs_m2]).drop_duplicates()
            df_final_probs = df_final_probs.sort_values('index')  # so that index of different models will appear together row-row

        df_final_features = new_dataframe_prep_based_on_effective_index(df_features, df_final_probs)

    elif detected_single_yaxis(relayout_data):
        y_start_idx = relayout_data['yaxis.range[0]']
        y_stop_idx = relayout_data['yaxis.range[1]']

        condition_min_loss = dfs_viz[0]['lloss'] >= y_start_idx
        condition_max_loss = dfs_viz[0]['lloss'] <= y_stop_idx
        df_final_probs = conditional_sliced_df(dfs_viz[0], condition_min_loss, condition_max_loss)

        if detected_bimodal(models):
            condition_min_loss_m2 = dfs_viz[1]['lloss'] >= y_start_idx
            condition_max_loss_m2 = dfs_viz[1]['lloss'] <= y_stop_idx
            df_final_probs_m2 = conditional_sliced_df(dfs_viz[1], condition_min_loss_m2, condition_max_loss_m2)

            df_final_probs = pd.concat([df_final_probs, df_final_probs_m2]).drop_duplicates()
            df_final_probs = df_final_probs.sort_values('index')  # so that index of different models will appear together row-row

        df_final_features = new_dataframe_prep_based_on_effective_index(df_features, df_final_probs)

    else:  
        '''
        detected_single_xaxis or a complete range is provided by user (with proper x-y coordinates) 
        will have same results due to the setup of dfs_viz for cls (loss values are tight to cluster group)
        '''
        x_cluster = get_effective_xaxis_cluster(relayout_data)
        df_filtered_x = dfs_viz[0][dfs_viz[0]['cluster'] == x_cluster]

        y_start_idx = relayout_data['yaxis.range[0]']
        y_stop_idx = relayout_data['yaxis.range[1]']
        condition_min_loss = df_filtered_x['lloss'] >= y_start_idx
        condition_max_loss = df_filtered_x['lloss'] <= y_stop_idx
        df_final_probs = conditional_sliced_df(df_filtered_x, condition_min_loss, condition_max_loss)

        if detected_bimodal(models):
            df_filtered_x_m2 = dfs_viz[1][dfs_viz[1]['cluster'] == x_cluster]
            condition_min_loss_m2 = df_filtered_x_m2['lloss'] >= y_start_idx
            condition_max_loss_m2 = df_filtered_x_m2['lloss'] <= y_stop_idx
            df_final_probs_m2 = conditional_sliced_df(df_filtered_x_m2, condition_min_loss_m2, condition_max_loss_m2)

            df_final_probs = pd.concat([df_final_probs, df_final_probs_m2]).drop_duplicates()
            df_final_probs = df_final_probs.sort_values('index')  # so that index of different models will appear together row-row

        df_final_features = new_dataframe_prep_based_on_effective_index(df_features, df_final_probs)
    return df_final_features, df_final_probs


def _display_score(ls_cluster_score: List[float], models: List[str]):
    '''
    Internal function to tap-out text field for silhouette score
    '''
    score_text = f'Silhouette score: {ls_cluster_score[0]}'
    if detected_bimodal(models):
        score_text = f'Silhouette score: {ls_cluster_score[0]} [ {models[0]} ] ' \
                    f'{ls_cluster_score[1]} [ {models[1]} ]'
    return score_text


class LossClusters:
    '''
    Main integration for feature component on Loss Clusters.

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Important Attributes:

        analysis_type (str):
            Analysis type defined by user during initial inputs preparation via data_loader stage.
        model_names (:obj:`List[str]`):
            model names defined by user during initial inputs preparation via data_loader stage.
        is_bimodal (bool):
            to indicate if analysis involves 2 models
        num_clusters (int):
            Number of cluster to form
        log_funct (:obj:`math.log`):
            Mathematics logarithm function used to calculate log-loss between yTrue and yPred
        specific_dataset (str):
            Default to 'All' indicating to include all miss-predict labels. Other options flexibly expand depending on class labels

    Returns:
        :obj:`~dash_core_components.Container`:
            styled dash components displaying graph and/or table objects
    '''
    def __init__(self, data_loader: Union[CSVDataLoader, DataframeLoader]):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.model_names = data_loader.get_model_list()
        self.is_bimodal = True if len(self.model_names) > 1 else False
        self.num_cluster = 4

        # instantiate at this stage due to shared use with callbacks
        if is_regression(self.analysis_type):
            self.compact_outputs_reg = fig_plot_offset_clusters_reg(self.data_loader, self.num_cluster)
            self.df, self.offset_clusters_reg = self.compact_outputs_reg[0], self.compact_outputs_reg[1]
            self.ls_cluster_score, self.optimum_elbow_reg = self.compact_outputs_reg[2], self.compact_outputs_reg[3]
            self.cols_table_reg = [col.replace('_', ' ') for col in self.df.columns]
            self.score_text = _display_score(self.ls_cluster_score, self.model_names)

        elif is_classification(self.analysis_type):
            self.log_func = math.log
            self.specific_dataset = 'All'

            self.compact_outputs_cls = fig_plot_logloss_clusters_cls(self.data_loader, self.num_cluster, self.log_func, self.specific_dataset)
            self.ls_dfs_viz, self.lloss_clusters_cls, = self.compact_outputs_cls[0], self.compact_outputs_cls[1]
            self.ls_cluster_score, self.optimum_elbow_cls = self.compact_outputs_cls[2], self.compact_outputs_cls[3]
            self.ls_class_labels, self.ls_class_labels_misspred = self.compact_outputs_cls[4], self.compact_outputs_cls[5]
            self.score_text = _display_score(self.ls_cluster_score, self.model_names)

    def show(self):
        '''
        Method to tapout styled html for loss clusters
        '''
        if is_regression(self.analysis_type):
            lloss_clusters = dbc.Container([
                                    dbc.Row(html.Div(
                                        html.H5('Optimum Cluster via Elbow Method', className='h5__cluster-section-title'))),
                                    dbc.Row(
                                        dcc.Graph(id='fig-optimum-cluster-reg',
                                                figure=self.optimum_elbow_reg,),
                                        justify='center', className='border__optimum-cluster'),
                                    dbc.Row(html.H5('Log-Loss Clustering via KMean', className='h5__cluster-section-title')),
                                    dbc.Row([
                                            dbc.Col([
                                                dbc.Row(html.Div(html.H6('Select No. of Cluster'), className='h6__cluster-instruction')),
                                                dbc.Row(dbc.Select(id='select-num-cluster-reg',
                                                            options=style_configs.OPTIONS_NO_OF_CLUSTERS,
                                                            value='4'), className='params__select-cluster')
                                            ], width=6),
                                            dbc.Col(width=4),
                                            dbc.Col(
                                                dbc.Row(dcc.Loading(id='loading-output-loss-cluster-reg',
                                                                    type='circle', color='#a80202'),
                                                        justify='right', className='loading__loss-cluster'), width=1),
                                            dbc.Col(
                                                dbc.Row(dbc.Button("Update",
                                                                    id='button-num-cluster-update-reg',
                                                                    n_clicks=0,
                                                                    color="info", 
                                                                    className='button__update-dataset'),
                                                        justify='right'))], className='border__select-dataset'),
                                    dbc.Row(dbc.Col(dbc.Row(
                                        html.Div(self.score_text,
                                                id='text-score-cluster-reg',
                                                className='text__score-cluster-reg'), justify='right'))),
                                    dbc.Row(
                                        dcc.Graph(id='fig-loss-cluster-reg',
                                                    figure=self.offset_clusters_reg),
                                        justify='center', className='border__common-cluster-plot-reg'),

                                    html.Div(html.H6(style_configs.INSTRUCTION_TEXT_SHARED), className='h6__dash-table-instruction-cluster-reg'),
                                    html.Div(id='alert-to-reset-cluster-reg'),
                                    html.Div(id='table-feat-prob-cluster-reg', className='div__table-proba-misspred'),
                                    html.Br()], fluid=True)
            return lloss_clusters

        elif is_classification(self.analysis_type):
            options_misspred_dataset = [{'label': 'All', 'value': 'All'}] + \
                                        [{'label': f'class {label}', 'value': f'class {label}'} for label in self.ls_class_labels_misspred]

            lloss_clusters = dbc.Container([
                                    dbc.Row(html.H5('Loss Cluster Analysis for ALL Miss Predictions',
                                        id='title-after-misspred-dataset-selection-cls',
                                        className='h5__cluster-section-title')),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Row(html.Div(html.H6('Select Miss-Predict Dataset'), className='h6__cluster-instruction')),
                                            dbc.Row(dbc.Select(id='select-misspred-dataset-cls',
                                                        options=options_misspred_dataset,
                                                        value='All'), className='params__select-cluster')
                                        ], width=4),
                                        dbc.Col(width=6),
                                        dbc.Col(
                                                dbc.Row(dcc.Loading(id='loading-output-misspred-dataset-cls',
                                                                    type='circle', color='#a80202'),
                                                        justify='right', className='loading__loss-cluster'), width=1),
                                        dbc.Col(
                                            dbc.Row(dbc.Button("Update",
                                                                id='button-misspred-dataset-update-cls',
                                                                n_clicks=0,
                                                                color="info",
                                                                className='button__update-dataset'),
                                                    justify='right'))], className='border__select-dataset'),
                                    html.Div(id='alert-clustering-error-cls'),
                                    dbc.Row(dcc.Graph(id='fig-cls-optimum-cluster', figure=self.optimum_elbow_cls,),
                                        justify='center', className='border__optimum-cluster'),
                                    dbc.Row(html.H5('Log-Loss Clustering via KMean on ALL Miss Predictions',
                                                    id='title-after-losscluster-params-selection-cls',
                                                    className='h5__cluster-section-title-kmean')),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Row(html.Div(html.H6('Select No. of Cluster'), className='h6__cluster-instruction')),
                                            dbc.Row(dbc.Select(id='select-num-cluster-cls',
                                                        options=style_configs.OPTIONS_NO_OF_CLUSTERS,
                                                        value='4'), className='params__select-cluster')
                                        ], width=4),
                                        dbc.Col([
                                            dbc.Row(html.Div(html.H6('Select Logarithm Method'), className='h6__cluster-instruction')),
                                            dbc.Row(dbc.Select(id='select-log-method-cls',
                                                        options=[{'label': 'LOG', 'value': 'log'},
                                                                {'label': 'LOG1P', 'value': 'log1p'},
                                                                {'label': 'LOG2', 'value': 'log2'},
                                                                {'label': 'LOG10', 'value': 'log10'}],
                                                        value='log'), className='params__select-cluster')
                                        ], width=4),
                                        dbc.Col(width=2),
                                        dbc.Col(
                                                dbc.Row(dcc.Loading(id='loading-output-loss-cluster-cls',
                                                                    type='circle', color='#a80202'),
                                                        justify='right', className='loading__loss-cluster'), width=1),
                                        dbc.Col(
                                            dbc.Row(dbc.Button("Update",
                                                                id='button-logloss-update-cls',
                                                                n_clicks=0,
                                                                color="info",
                                                                className='button__update-dataset'),
                                                    justify='right'))], className='border__select-dataset'),
                                    dbc.Row(dbc.Col(dbc.Row(
                                        html.Div(self.score_text,
                                                id='text-score-cluster-cls',
                                                className='text__score-cluster-cls'), justify='right'))),
                                    dbc.Row(
                                        dcc.Graph(id='fig-loss-cluster-cls',
                                                    figure=self.lloss_clusters_cls,),
                                        justify='center', className='border__common-cluster-plot-cls'),

                                    # data-table, appeared only after data range selection on fig-loss-cluster-cls by user
                                    html.Div(html.H6(style_configs.INSTRUCTION_TEXT_SHARED), className='h6__dash-table-instruction-cls'),
                                    html.Div(id='alert-to-reset-loss-cluster-cls'),
                                    html.Div(id='table-title-features-loss-cluster'),
                                    html.Div(id='show-feat-table-loss-cluster', className='div__table-proba-misspred'),
                                    html.Br(),
                                    html.Div(id='table-title-probs-loss-cluster'),
                                    html.Div(id='show-prob-table-loss-cluster', className='div__table-proba-misspred'),
                                    html.Br()], fluid=True)
            return lloss_clusters

    def callbacks(self):
        @app.callback(
            Output('loading-output-loss-cluster-reg', 'children'),
            Output('text-score-cluster-reg', 'children'),
            Output('fig-loss-cluster-reg', 'figure'),
            Input('button-num-cluster-update-reg', 'n_clicks'),
            State('select-num-cluster-reg', 'value'))
        def update_fig_based_on_selected_num_cluster(click_count, selected_no_cluster):
            '''
            Callbacks functionalities specific to param - select no. of clusters [ regression ]
            '''
            if click_count > 0:
                _, fig_obj_cluster_reg, ls_cluster_score_reg, _ = fig_plot_offset_clusters_reg(self.data_loader, int(selected_no_cluster))
                score_text_reg = _display_score(ls_cluster_score_reg, self.model_names)
                return '', score_text_reg, fig_obj_cluster_reg
            else:
                raise PreventUpdate

        @app.callback(
            Output('alert-to-reset-cluster-reg', 'children'),
            Output('table-feat-prob-cluster-reg', 'children'),
            Input('fig-loss-cluster-reg', 'relayoutData'),
            Input('fig-loss-cluster-reg', 'restyleData'),
            State('select-num-cluster-reg', 'value'))
        def display_table_based_on_selected_range_reg(relayout_data, restyle_data, selected_no_cluster):
            '''
            Callbacks functionalities specific to reponse from fig-obj to data-table [ regression ]
            '''
            if relayout_data is not None:
                df_usr_select_cluster, _, _, _ = fig_plot_offset_clusters_reg(self.data_loader, int(selected_no_cluster))
                try:
                    # to limit table cell having values with long decimals for better viz purpose
                    df_usr_select_cluster = df_usr_select_cluster.round(2)
                except TypeError:
                    df_usr_select_cluster

                if is_reset(relayout_data):
                    alert_obj_reg = None
                    table_obj_reg = None

                elif is_active_trace(relayout_data):
                    models = self.model_names
                    if restyle_data is not None:  # [{'visible': ['legendonly']}, [1]]
                        if detected_legend_filtration(restyle_data):
                            model_to_exclude_from_view = self.model_names[restyle_data[1][0]]
                            models = [model for model in self.model_names if model != model_to_exclude_from_view]

                    default_header = style_configs.default_header_style()
                    alert_obj_reg = style_configs.activate_alert()

                    df_final = convert_cluster_relayout_data_to_df_reg(relayout_data, df_usr_select_cluster, models)
                    df_final.columns = self.cols_table_reg  # to have customized column names displayed on table

                    data_relayout_reg = df_final.to_dict('records')
                    table_obj_reg = table_with_relayout_datapoints(data_relayout_reg, self.cols_table_reg, default_header, 'csv')
                return alert_obj_reg, table_obj_reg
            else:
                raise PreventUpdate

        @app.callback(
            Output('loading-output-misspred-dataset-cls', 'children'),
            Output('loading-output-loss-cluster-cls', 'children'),
            Output('alert-clustering-error-cls', 'children'),
            Output('title-after-misspred-dataset-selection-cls', 'children'),
            Output('fig-cls-optimum-cluster', 'figure'),
            Output('title-after-losscluster-params-selection-cls', 'children'),
            Output('text-score-cluster-cls', 'children'),
            Output('fig-loss-cluster-cls', 'figure'),
            Input('button-misspred-dataset-update-cls', 'n_clicks'),
            Input('button-logloss-update-cls', 'n_clicks'),
            State('select-misspred-dataset-cls', 'value'),
            State('select-num-cluster-cls', 'value'),
            State('select-log-method-cls', 'value'))
        def update_loss_cluster_tab_based_on_selected_misspred_dataset(click_count_dataset,
                                                                        click_count_params,
                                                                        selected_dataset,
                                                                        selected_cluster,
                                                                        selected_method):
            '''
            Callbacks functionalities specific to all params selection [ classification ]
            '''
            ctx = dash.callback_context
            triggered_button = ctx.triggered[0]['prop_id'].split('.')[0]
            triggered_button_value = ctx.triggered[0]['value']

            current_dataset_name = ctx.states['select-misspred-dataset-cls.value']
            specific_dataset = current_dataset_name.replace('class ', '') if 'class' in current_dataset_name else current_dataset_name
            cluster_err_alert = style_configs.no_error_alert()

            # for click action on dataset selection
            if (triggered_button == 'button-misspred-dataset-update-cls') and (triggered_button_value > 0):
                title_aft_misspred_dataset = f'Loss Cluster Analysis for {selected_dataset.capitalize()} Miss Predictions'
                title_aft_params = f'Log-Loss Clustering via KMean on {selected_dataset.capitalize()} Miss Predictions'

                # pre-requisite to check if dataset is valid with sufficient data-points for auto-clustering
                ls_dfs_prob_misspred = IntLossClusterer(self.data_loader).extract_misspredictions()
                if specific_dataset != 'All' and any(len(df[df['yPred-label'] == specific_dataset]) < 8 for df in ls_dfs_prob_misspred):
                    cluster_err_alert = style_configs.activate_cluster_error_alert(specific_dataset)
                    return dash.no_update, dash.no_update, cluster_err_alert, dash.no_update, \
                            dash.no_update, dash.no_update, dash.no_update, dash.no_update

                outputs_callback_dataset = fig_plot_logloss_clusters_cls(self.data_loader,
                                                                        num_cluster=int(selected_cluster),
                                                                        log_func=style_configs.LOG_METHOD_DICT[selected_method],
                                                                        specific_dataset=specific_dataset)

                fig_obj_cluster_cls, ls_cluster_score_cls = outputs_callback_dataset[1], outputs_callback_dataset[2]
                fig_obj_elbow_cls = outputs_callback_dataset[3]
                text_score_cls = _display_score(ls_cluster_score_cls, self.model_names)
                return '', dash.no_update, cluster_err_alert, title_aft_misspred_dataset, fig_obj_elbow_cls, \
                        title_aft_params, text_score_cls, fig_obj_cluster_cls

            # for click action on num_cluster and log_method selection
            elif (triggered_button == 'button-logloss-update-cls') and (triggered_button_value > 0):
                outputs_callback_params = fig_plot_logloss_clusters_cls(self.data_loader,
                                                                        num_cluster=int(selected_cluster),
                                                                        log_func=style_configs.LOG_METHOD_DICT[selected_method],
                                                                        specific_dataset=specific_dataset)

                fig_obj_cluster_cls_params, ls_cluster_score_cls_params = outputs_callback_params[1], outputs_callback_params[2]
                text_score_cls_params = _display_score(ls_cluster_score_cls_params, self.model_names)
                return dash.no_update, '', cluster_err_alert, dash.no_update, dash.no_update, dash.no_update, \
                        text_score_cls_params, fig_obj_cluster_cls_params

            else:
                raise PreventUpdate

        @app.callback(
            Output('alert-to-reset-loss-cluster-cls', 'children'),
            Output('table-title-features-loss-cluster', 'children'),
            Output('show-feat-table-loss-cluster', 'children'),
            Output('table-title-probs-loss-cluster', 'children'),
            Output('show-prob-table-loss-cluster', 'children'),
            Input('fig-loss-cluster-cls', 'relayoutData'),
            Input('fig-loss-cluster-cls', 'restyleData'),
            State('select-misspred-dataset-cls', 'value'),
            State('select-num-cluster-cls', 'value'),
            State('select-log-method-cls', 'value'))
        def display_table_based_on_selected_range_cls(relayout_data, restyle_data, selected_dataset, selected_cluster, selected_method):
            '''
            Callbacks functionalities specific to reponse from fig-obj to data-table [ classification ]
            '''
            default_title = style_configs.DEFAULT_TITLE_STYLE
            title_table_features_cls = html.H6('Feature Values :', style=default_title, className='title__table-misspred-cls')
            title_table_probs_cls = html.H6('Probabilities Overview :', style=default_title, className='title__table-misspred-cls')

            if relayout_data is not None:
                specific_dataset = selected_dataset.replace('class ', '') if 'class' in selected_dataset else selected_dataset
                outputs_callback_fig_action = fig_plot_logloss_clusters_cls(self.data_loader,
                                                                        num_cluster=int(selected_cluster),
                                                                        log_func=style_configs.LOG_METHOD_DICT[selected_method],
                                                                        specific_dataset=specific_dataset)
                dfs_viz, df_features = outputs_callback_fig_action[0], outputs_callback_fig_action[6]

                try:
                    df_features = df_features.round(2)  # limit long decimals on feature values
                    dfs_viz[0] = dfs_viz[0].round(4)  # standardize prob values to 4 decimals
                    if self.is_bimodal:
                        dfs_viz[1] = dfs_viz[1].round(4)
                except TypeError:
                    df_features
                    dfs_viz

                df_features = insert_index_col(df_features)
                dfs_viz = [insert_index_col(df) for df in dfs_viz]

                if is_reset(relayout_data):
                    alert_obj_cls = None
                    title_table_features_cls = None
                    table_obj_features_cls = None
                    title_table_probs_cls = None
                    table_obj_probs_cls = None

                elif is_active_trace(relayout_data):
                    models = self.model_names
                    if restyle_data is not None:  # [{'visible': ['legendonly']}, [1]]
                        if detected_legend_filtration(restyle_data):
                            model_to_exclude_from_view = self.model_names[restyle_data[1][0]]
                            models = [model for model in self.model_names if model != model_to_exclude_from_view]

                    default_header = style_configs.default_header_style()
                    alert_obj_cls = style_configs.activate_alert()

                    # dfs_viz adjusted to the correct df according to the filtered model following click action on legend
                    dfs_viz_adjusted = get_adjusted_dfs_based_on_legend_filtration(dfs_viz, models)
                    df_final_features, df_final_probs = convert_cluster_relayout_data_to_df_cls(relayout_data,
                                                                                                dfs_viz_adjusted,
                                                                                                df_features,
                                                                                                models)

                    data_relayout_features_cls = df_final_features.to_dict('records')
                    data_relayout_prob_cls = df_final_probs.to_dict('recorfs')
                    table_obj_features_cls = table_with_relayout_datapoints(data_relayout_features_cls,
                                                                            df_final_features.columns,
                                                                            default_header,
                                                                            'csv')
                    table_obj_probs_cls = table_with_relayout_datapoints(data_relayout_prob_cls,
                                                                        df_final_probs.columns,
                                                                        default_header,
                                                                        'csv')
                return alert_obj_cls, title_table_features_cls, table_obj_features_cls, title_table_probs_cls, table_obj_probs_cls
            else:
                raise PreventUpdate

import math
import pandas as pd

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from tenjin.app import app
from tenjin.interpreters.structured_data import IntLossClusterer
from tenjin.visualizers import loss_clusters as viz_clusters
from tenjin.visualizers import shared_viz_component as viz_shared
from tenjin.utils import style_configs
from tenjin.utils.common_functions import (is_active_trace, is_reset, detected_legend_filtration, detected_single_xaxis, detected_single_yaxis,
                                        detected_bimodal, get_min_max_offset, get_min_max_cluster, get_effective_xaxis_cluster,
                                        conditional_sliced_df, dataframe_prep_on_model_count_by_yaxis_slice)


OPTIONS_NO_OF_CLUSTERS = [{'label': f'{n}', 'value': f'{n}'} for n in range(2, 9)]  # option 2 to 8


def fig_plot_offset_clusters_reg(data_loader, num_cluster):
    df, ls_cluster_score, analysis_type, ls_cluster_range, ls_ssd = IntLossClusterer(data_loader).xform(num_cluster, None, 'All')
    models = data_loader.get_model_list()

    fig_obj_cluster = viz_clusters.plot_offset_clusters(df, analysis_type)
    fig_obj_elbow = viz_clusters.plot_optimum_cluster_via_elbow_method(ls_cluster_range, ls_ssd, models)
    return df, fig_obj_cluster, ls_cluster_score, fig_obj_elbow


def fig_plot_logloss_clusters_cls(data_loader, num_cluster, log_func=math.log, specific_label='All'):
    compact_outputs = IntLossClusterer(data_loader).xform(num_cluster, log_func, specific_label)
    ls_dfs_viz, ls_class_labels, ls_class_labels_misspred = compact_outputs[0], compact_outputs[1], compact_outputs[2]
    ls_cluster_score, analysis_type = compact_outputs[3], compact_outputs[4]
    ls_cluster_range, ls_ssd = compact_outputs[5], compact_outputs[6]

    models = data_loader.get_model_list()
    fig_obj_cluster = viz_clusters.plot_logloss_clusters(ls_dfs_viz, analysis_type)
    fig_obj_elbow = viz_clusters.plot_optimum_cluster_via_elbow_method(ls_cluster_range, ls_ssd, models)
    return ls_dfs_viz, fig_obj_cluster, ls_cluster_score, fig_obj_elbow, ls_class_labels, ls_class_labels_misspred


def table_with_relayout_datapoints(data, customized_cols, header, exp_format):
    tab_obj = viz_shared.reponsive_table_to_filtered_datapoints(data, customized_cols, header, exp_format)
    return tab_obj


def convert_cluster_relayout_data_to_df_reg(relayout_data, df, models):
    """convert raw data format from relayout selection range by user into the correct df fit for viz purpose

    Arguments:
        relayout_data {dict}: data containing selection range indices returned from plotly graph
        df {pandas dataframe}: dataframe tap-out from interpreters pipeline
        models {list}: model names defined by user during spin-up of Tenjin app

    Returns:
        pandas dataframe
        -- dataframe fit for the responsive table-graph filtering
    """
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


def _display_score(ls_cluster_score, models):
    score_text = f'Silhouette score: {ls_cluster_score[0]}'
    if len(models) == 2:
        score_text = f'Silhouette score: {ls_cluster_score[0]} [ {models[0]} ] ' \
                    f'{ls_cluster_score[1]} [ {models[1]} ]'
    return score_text


class LossClusters:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.model_names = data_loader.get_model_list()
        self.is_bimodal = True if len(self.model_names) > 1 else False
        self.num_cluster = 4

        if self.analysis_type == 'regression':
            self.compact_outputs_reg = fig_plot_offset_clusters_reg(self.data_loader, self.num_cluster)
            self.df, self.offset_clusters_reg = self.compact_outputs_reg[0], self.compact_outputs_reg[1]
            self.ls_cluster_score, self.optimum_elbow_reg = self.compact_outputs_reg[2], self.compact_outputs_reg[3]
            self.cols_table_reg = [col.replace('_', ' ') for col in self.df.columns]
            self.score_text = _display_score(self.ls_cluster_score, self.model_names)

        elif 'classification' in self.analysis_type:
            self.log_func = math.log
            self.specific_label = 'All'

            self.compact_outputs_cls = fig_plot_logloss_clusters_cls(self.data_loader, self.num_cluster, self.log_func, self.specific_label)
            self.ls_dfs_viz, self.lloss_clusters_cls, = self.compact_outputs_cls[0], self.compact_outputs_cls[1]
            self.ls_cluster_score, self.optimum_elbow_cls = self.compact_outputs_cls[2], self.compact_outputs_cls[3]
            self.ls_class_labels, self.ls_class_labels_misspred = self.compact_outputs_cls[4], self.compact_outputs_cls[5]

            self.score_text = f'Silhouette score: {self.ls_cluster_score[0]}'
            if self.is_bimodal:
                self.score_text = f'Silhouette score: {self.ls_cluster_score[0]} [ {self.model_names[0]} ] ' \
                            f'{self.ls_cluster_score[1]} [ {self.model_names[1]} ]'

    def show(self):
        if self.analysis_type == 'regression':
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
                                                            options=OPTIONS_NO_OF_CLUSTERS,
                                                            value='4'), className='params__select-cluster')
                                            ], width=6),
                                            dbc.Col(
                                                dbc.Row(dbc.Button("Update",
                                                                    id='button-select-num-cluster-reg',
                                                                    n_clicks=0,
                                                                    color="info", 
                                                                    className='button__update-dataset'),
                                                        justify='right'), width=1),
                                            dbc.Col(
                                                dbc.Row(dcc.Loading(id='loading-output-loss-cluster-reg',
                                                                    type='circle', color='#a80202'),
                                                        justify='left', className='loading__loss-cluster-reg'), width=1)],
                                        className='border__select-dataset'),
                                    dbc.Row(dbc.Col(dbc.Row(
                                        html.Div(self.score_text,
                                                id='text-score-cluster-reg',
                                                className='text__score-cluster-reg'), justify='right'))),
                                    dbc.Row(
                                        dcc.Graph(id='fig-loss-cluster-reg',
                                                    figure=self.offset_clusters_reg),
                                        justify='center', className='border__common-cluster-plot-reg'),

                                    html.Div(id='alert-to-reset-cluster-reg'),
                                    html.Div(id='table-feat-prob-cluster-reg', className='div__table-proba-misspred'),
                                    html.Br()], fluid=True)
            return lloss_clusters

        elif 'classification' in self.analysis_type:
            options_misspred_dataset = [{'label': 'All', 'value': 'All'}] + \
                                        [{'label': f'class {label}', 'value': f'class {label}'} for label in self.ls_class_labels_misspred]
            lloss_clusters = dbc.Container([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Row(html.Div(html.H6('Select Miss-Predict Dataset'), className='h6__cluster-instruction')),
                                            dbc.Row(dbc.Select(id='select-misspred-dataset-cls',
                                                        options=options_misspred_dataset,
                                                        value='All'), className='params__select-cluster')
                                        ], width=6),
                                        dbc.Col(
                                            dbc.Row(dbc.Button("Update",
                                                                id='button__logloss-update-cls',
                                                                n_clicks=0,
                                                                color="info",
                                                                className='button__update-dataset'),
                                                    justify='right'))], className='border__select-dataset'),
                                    dbc.Row(dcc.Graph(id='fig-reg-optimum-cluster', figure=self.optimum_elbow_cls,),
                                        justify='center', className='border__optimum-cluster'),
                                    dbc.Row(html.H5('Log-Loss Clustering via KMean', className='h5__cluster-section-title')),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Row(html.Div(html.H6('Select No. of Cluster'), className='h6__cluster-instruction')),
                                            dbc.Row(dbc.Select(id='select-num-cluster-cls',
                                                        options=OPTIONS_NO_OF_CLUSTERS,
                                                        value='4'), className='params__select-cluster')
                                        ], width=5),
                                        dbc.Col([
                                            dbc.Row(html.Div(html.H6('Select Logarithm Method'), className='h6__cluster-instruction')),
                                            dbc.Row(dbc.Select(id='select-num-cluster-cls',
                                                        options=[{'label': 'LOG', 'value': 'log'},
                                                                {'label': 'LOG1P', 'value': 'log1p'},
                                                                {'label': 'LOG2', 'value': 'log2'},
                                                                {'label': 'LOG10', 'value': 'log10'}],
                                                        value='log'), className='params__select-cluster')
                                        ], width=5),
                                        dbc.Col(
                                            dbc.Row(dbc.Button("Update",
                                                                id='button__logloss-update-cls',
                                                                n_clicks=0,
                                                                color="info",
                                                                className='button__update-dataset'),
                                                    justify='right'), width=2)], className='border__select-dataset'),
                                    dbc.Row(dbc.Col(dbc.Row(
                                        html.Div(self.score_text,
                                                id='score__cluster-reg',
                                                className='text__score-cluster-cls'), justify='right'))),
                                    dbc.Row(
                                        dcc.Graph(id='fig-reg-lloss-clusters',
                                                    figure=self.lloss_clusters_cls,),
                                        justify='center', className='border__common-cluster-plot-cls'),
                                    html.Br()], fluid=True)
            return lloss_clusters

    def callbacks(self):
        @app.callback(
            Output('loading-output-loss-cluster-reg', 'children'),
            Output('text-score-cluster-reg', 'children'),
            Output('fig-loss-cluster-reg', 'figure'),
            Input('button-select-num-cluster-reg', 'n_clicks'),
            State('select-num-cluster-reg', 'value'))
        def update_fig_based_on_selected_num_cluster(click_count, selected_no_cluster):
            print(f'click_count: {click_count}')
            if click_count > 0:
                print(f'selected_no_cluster: {selected_no_cluster}')
                _, fig_obj_cluster_usr, ls_cluster_score_usr, _ = fig_plot_offset_clusters_reg(self.data_loader, int(selected_no_cluster))
                score_text_usr = _display_score(ls_cluster_score_usr, self.model_names)
                return '', score_text_usr, fig_obj_cluster_usr
            else:
                raise PreventUpdate

        @app.callback(
            Output('alert-to-reset-cluster-reg', 'children'),
            Output('table-feat-prob-cluster-reg', 'children'),
            Input('fig-loss-cluster-reg', 'relayoutData'),
            Input('fig-loss-cluster-reg', 'restyleData'),
            State('select-num-cluster-reg', 'value'))
        def display_table_based_on_selected_range_reg(relayout_data, restyle_data, selected_no_cluster):
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

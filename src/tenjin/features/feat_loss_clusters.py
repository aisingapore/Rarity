import math
from dash.dependencies import Input, Output, ALL, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from tenjin.app import app
from tenjin.interpreters.structured_data import IntLossClusterer
from tenjin.visualizers import loss_clusters as viz_clusters
from tenjin.visualizers import shared_viz_component as viz_shared
from tenjin.utils.common_function import (convert_relayout_data_to_df_reg, convert_relayout_data_to_df_cls, identify_active_trace,
                                        is_active_trace, is_reset, detected_legend_filtration, detected_unique_figure,
                                        detected_more_than_1_unique_figure, insert_index_col)


def fig_plot_offset_clusters_reg(data_loader, num_cluster=4):
    df, ls_cluster_score, analysis_type, ls_cluster_range, ls_ssd = IntLossClusterer(data_loader).xform(num_cluster, None)
    models = data_loader.get_model_list()

    fig_obj_cluster = viz_clusters.plot_offset_clusters(df, analysis_type)
    fig_obj_elbow = viz_clusters.plot_optimum_cluster_via_elbow_method(ls_cluster_range, ls_ssd, models)
    return df, fig_obj_cluster, ls_cluster_score, fig_obj_elbow


def table_with_relayout_datapoints(data, customized_cols, header, exp_format):
    tab_obj = viz_shared.reponsive_table_to_filtered_datapoints(data, customized_cols, header, exp_format)
    return tab_obj


def fig_plot_logloss_clusters_cls(data_loader, num_cluster=4, log_func=math.log):
    compact_outputs = IntLossClusterer(data_loader).xform(num_cluster, log_func)
    ls_dfs_viz, ls_class_labels, ls_cluster_score = compact_outputs[0], compact_outputs[1], compact_outputs[2]
    analysis_type, ls_cluster_range, ls_ssd = compact_outputs[3], compact_outputs[4], compact_outputs[5]
    ls_dfs_misspred = [df[df['pred_state'] == 'miss-predict'] for df in ls_dfs_viz]
    models = data_loader.get_model_list()

    fig_obj_cluster = viz_clusters.plot_logloss_clusters(ls_dfs_misspred, analysis_type)
    fig_obj_elbow = viz_clusters.plot_optimum_cluster_via_elbow_method(ls_cluster_range, ls_ssd, models)
    return ls_dfs_misspred, fig_obj_cluster, ls_cluster_score, fig_obj_elbow, ls_class_labels


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

        if self.analysis_type == 'regression':
            self.df, self.offset_clusters_reg, self.ls_cluster_score, self.optimum_elbow_reg = fig_plot_offset_clusters_reg(self.data_loader)
            self.cols_table_reg = [col.replace('_', ' ') for col in self.df.columns]
            self.score_text = _display_score(self.ls_cluster_score, self.model_names)

        elif 'classification' in self.analysis_type:
            self.compact_outputs = fig_plot_logloss_clusters_cls(self.data_loader)
            self.ls_dfs_misspred, self.lloss_clusters_cls, = self.compact_outputs[0], self.compact_outputs[1]
            self.ls_cluster_score, self.optimum_elbow_cls = self.compact_outputs[2], self.compact_outputs[3]
            self.ls_class_labels = self.compact_outputs[4]
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
                                        dcc.Graph(id='fig-reg-optimum-cluster', 
                                                figure=self.optimum_elbow_reg,),
                                        justify='center', className='border__optimum-cluster'),
                                    dbc.Row(html.H5('Log-Loss Clustering via KMean', className='h5__cluster-section-title')),
                                    dbc.Row([
                                            dbc.Col([
                                                dbc.Row(html.Div(html.H6('Select No. of Cluster'), className='h6__cluster-instruction')),
                                                dbc.Row(dbc.Select(id='select-num-cluster-reg', 
                                                            options=[{'label': '2', 'value': '2'},
                                                                    {'label': '3', 'value': '3'},
                                                                    {'label': '4', 'value': '4'},
                                                                    {'label': '5', 'value': '5'},
                                                                    {'label': '6', 'value': '6'},
                                                                    {'label': '7', 'value': '7'},
                                                                    {'label': '8', 'value': '8'}],
                                                            value='4'), className='params__select-cluster')
                                            ], width=5),
                                            dbc.Col(
                                                dbc.Row(dbc.Button("Update",
                                                                    id='button-select-num-cluster-reg',
                                                                    n_clicks=0,
                                                                    color="info", 
                                                                    className='button__update-cluster'),
                                                        justify='right'), width=1),
                                            dbc.Col(
                                                dbc.Row(dcc.Loading(id='loading-update-loss-cluster-reg',
                                                                    type='circle', color='#a80202'),
                                                        justify='left', className='loading__loss-cluster-reg'), width=1)],
                                            # dbc.Col(
                                            #     dbc.Row(
                                            #         html.Div(children=self.score_text,
                                            #                 id='text-score-cluster-reg',
                                            #                 className='text__score-cluster-reg'), justify='right'))],
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
            lloss_clusters = dbc.Container([
                                    dbc.Row(html.Div(
                                        html.H5('Optimum Cluster via Elbow Method', className='h5__cluster-section-title'))),
                                    dbc.Row(
                                        dcc.Graph(id='fig-reg-optimum-cluster', 
                                                figure=self.optimum_elbow_cls,),
                                        justify='center', className='border__optimum-cluster-cls'),
                                    dbc.Row(html.H5('Log-Loss Clustering via KMean', className='h5__cluster-section-title')),
                                    dbc.Row([
                                            dbc.Col([
                                                dbc.Row(html.Div(html.H6('Select No. of Cluster'), className='h6__cluster-instruction')),
                                                dbc.Row(dbc.Select(id='select-num-cluster-cls', 
                                                            options=[{'label': '2', 'value': '2'},
                                                                    {'label': '3', 'value': '3'},
                                                                    {'label': '4', 'value': '4'},
                                                                    {'label': '5', 'value': '5'},
                                                                    {'label': '6', 'value': '6'},
                                                                    {'label': '7', 'value': '7'},
                                                                    {'label': '8', 'value': '8'}],
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
                                                                    className='button__update-cluster'),
                                                        justify='right'), width=2)]),

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
            Output('loading-update-loss-cluster-reg', 'children'),
            Output('text-score-cluster-reg', 'children'),
            Output('fig-loss-cluster-reg', 'figure'),
            Input('button-select-num-cluster-reg', 'n_clicks'),
            State('select-num-cluster-reg', 'value'))
        def update_fig_based_on_selected_num_cluster(click_count, selected_no_cluster):
            print(f'click_count: {click_count}')
            if click_count > 0:
                print(f'selected_no_cluster: {selected_no_cluster}')
                # print(f'type_no_cluster: {type(selected_no_cluster)}')
                _, fig_obj_cluster_usr, ls_cluster_score_usr, _ = fig_plot_offset_clusters_reg(self.data_loader, int(selected_no_cluster))
                score_text_usr = _display_score(ls_cluster_score_usr, self.model_names)
                return '', score_text_usr, fig_obj_cluster_usr
            else:
                raise PreventUpdate

        @app.callback(
            Output('alert-to-reset-cluster-reg', 'children'),
            Output('table-feat-prob-cluster-reg', 'children'),
            Input({"index": ALL, "type": "fig-obj-prob-spread"}, 'relayoutData'),
            Input({"index": ALL, "type": "fig-obj-prob-spread"}, 'restyleData'),
            State('select-num-cluster-reg', 'value'))
        def display_table_based_on_selected_range_reg(relayout_data, restyle_data, selected_no_cluster):
            print(f'relayout_data: {relayout_data}')
            print(f'restyle_data: {restyle_data}')
            if relayout_data is not None:
                print(f'selected_no_cluster: {selected_no_cluster}')
                df_usr_select_cluster, _, _, _ = fig_plot_offset_clusters_reg(self.data_loader, int(selected_no_cluster))
                print(f'df_usr_select_cluster: {df_usr_select_cluster}')
                return '', ''
            else:
                raise PreventUpdate

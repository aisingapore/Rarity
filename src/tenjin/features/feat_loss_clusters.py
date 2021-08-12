import math
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from tenjin.interpreters.structured_data import IntLossClusterer
from tenjin.visualizers import loss_clusters as viz_clusters


def fig_plot_offset_clusters_reg(data_loader, num_cluster=4):
    df, ls_cluster_score, analysis_type, ls_cluster_range, ls_ssd = IntLossClusterer(data_loader).xform(num_cluster, None)
    models = data_loader.get_model_list()

    fig_obj_cluster = viz_clusters.plot_offset_clusters(df, analysis_type)
    fig_obj_elbow = viz_clusters.plot_optimum_cluster_via_elbow_method(ls_cluster_range, ls_ssd, models)
    return df, fig_obj_cluster, ls_cluster_score, fig_obj_elbow


def fig_plot_logloss_clusters_cls(data_loader, num_cluster=4, log_func=math.log):
    compact_outputs = IntLossClusterer(data_loader).xform(num_cluster, log_func)
    ls_dfs_viz, ls_class_labels, ls_cluster_score = compact_outputs[0], compact_outputs[1], compact_outputs[2]
    analysis_type, ls_cluster_range, ls_ssd = compact_outputs[3], compact_outputs[4], compact_outputs[5]
    ls_dfs_misspred = [df[df['pred_state'] == 'miss-predict'] for df in ls_dfs_viz]
    models = data_loader.get_model_list()

    fig_obj_cluster = viz_clusters.plot_logloss_clusters(ls_dfs_misspred, analysis_type)
    fig_obj_elbow = viz_clusters.plot_optimum_cluster_via_elbow_method(ls_cluster_range, ls_ssd, models)
    return ls_dfs_misspred, fig_obj_cluster, ls_cluster_score, fig_obj_elbow, ls_class_labels


class LossClusters:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.model_names = data_loader.get_model_list()
        self.is_bimodal = True if len(self.model_names) > 1 else False

        if self.analysis_type == 'regression':
            self.df, self.offset_clusters_reg, self.ls_cluster_score, self.optimum_elbow_reg = fig_plot_offset_clusters_reg(self.data_loader)
            self.score_text = f'Silhouette score: {self.ls_cluster_score[0]}'
            if self.is_bimodal:
                self.score_text = f'Silhouette score: {self.ls_cluster_score[0]} [ {self.model_names[0]} ] ' \
                            f'{self.ls_cluster_score[1]} [ {self.model_names[1]} ]'

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
                                                                    id='button__logloss-update-reg', 
                                                                    n_clicks=0,
                                                                    color="info", 
                                                                    className='button__update-cluster-reg'),
                                                        justify='right'), width=1),
                                            dbc.Col(
                                                dbc.Row(
                                                    html.Div(self.score_text,
                                                            id='score__cluster-reg',
                                                            className='text__score-cluster-reg'), justify='right'))]),
                                    dbc.Row(
                                        dcc.Graph(id='fig-reg-lloss-clusters',
                                                    figure=self.offset_clusters_reg,),
                                        justify='center', className='border__common-cluster-plot-reg'),
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

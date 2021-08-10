import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from tenjin.interpreters.structured_data import IntLossClusterer
from tenjin.visualizers import logloss_clusters as viz_clusters


def fig_plot_logloss_clusters_reg(data_loader, num_cluster=4):
    df, cluster_score_ls = IntLossClusterer(data_loader).xform(num_cluster)
    fig_obj = viz_clusters.plot_logloss_clusters(df, cluster_score_ls)
    return df, fig_obj, cluster_score_ls


class LogLossClusters:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.model_names = data_loader.get_model_list()
        self.is_bimodal = True if len(self.model_names) > 1 else False

        if self.analysis_type == 'regression':
            self.df, self.logloss_clusters_reg, self.cluster_score_ls = fig_plot_logloss_clusters_reg(self.data_loader)
            self.score_text = f'Silhouette score: {self.cluster_score_ls[0]}'
            if self.is_bimodal:
                self.score_text = f'Silhouette score: {self.cluster_score_ls[0]} [ {self.model_names[0]} ] ' \
                            f'{self.cluster_score_ls[1]} [ {self.model_names[1]} ]'

        elif 'classification' in self.analysis_type:
            pass

    def show(self):
        if self.analysis_type == 'regression':
            lloss_clusters = dbc.Container([
                                    dbc.Row([
                                            dbc.Col([
                                                dbc.Row(html.Div(html.H6('Select No. of Cluster'), className='h6__cluster-instruction-reg')),
                                                dbc.Row(dbc.Select(id='select-num-cluster-reg', 
                                                            options=[{'label': '2', 'value': '2'},
                                                                    {'label': '3', 'value': '3'},
                                                                    {'label': '4', 'value': '4'},
                                                                    {'label': '5', 'value': '5'},
                                                                    {'label': '6', 'value': '6'},
                                                                    {'label': '7', 'value': '7'},
                                                                    {'label': '8', 'value': '8'}],
                                                            value='4'))
                                            ], width=5, className='selection-bar-item__cluster-reg'),
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
                                                            className='text__score-cluster'), justify='right'))]),
                                    dbc.Row(
                                        dcc.Graph(id='fig-reg-lloss-clusters',
                                                    figure=self.logloss_clusters_reg,),
                                        justify='center', className='border__common-cluster-plot-reg'),
                                    html.Br()], fluid=True)
        return lloss_clusters

from typing import Union

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from rarity.data_loader import CSVDataLoader, DataframeLoader
from rarity.features import GeneralMetrics, MissPredictions, LossClusters, FeatureDistribution, SimilaritiesCF
from rarity.app import app


class GapAnalyzer:
    '''
    GapAnalyzer is the main class object collating all developed feature components for Rarity.
    Auto-generated error analysis supports single model and bimodal (max at comparison of 2 models side by side) on tasks
    such as regression, binary classification and multiclass classification

    Args:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            This is the class object from data_loader compiling ``xfeatures``, ``yTrue``, ``yPredict`` via either ``CSVDataLoader`` \
            (for both offline and inline analysis) or ``DataframeLoader`` (for inline analysis)
        user_defined_title (str):
            Title of analysis, text field defined by user

    Important Attributes:

        analysis_type (str):
            Analysis type defined by user. Corresponding feature components will be auto-populated based on the \
            specified analysis type. Supported analysis types : ``Regression``, ``Binary Classification``, ``Multiclass Classification``

    '''
    def __init__(self, data_loader: Union[CSVDataLoader, DataframeLoader], user_defined_title: str = None):
        self.data_loader = data_loader
        self.analysis_type = self.data_loader.get_analysis_type().replace('-', ' ').title()
        self.usr_defined_title = user_defined_title

    def _layout(self) -> dbc.Container:
        '''
        The main app layout of Rarity
        '''
        main_layout = dbc.Container([
                        dbc.Jumbotron(
                            dbc.Container([
                                dbc.Row([
                                    dbc.Col(html.Div([
                                        html.Img(className='header__rarity-logo', src='assets/rarity-icon.png'),
                                        html.H4('Gap Analysis with Rarity 1.0', className='header__rarity-title'),
                                        html.Pre(f'|  {self.usr_defined_title}', className='header__usr-pjt-title'),
                                    ]), width=8, md=8, sm=9, className='header__first-row-col-left align-self-center'),

                                    dbc.Col(html.Div(
                                        html.H4(self.analysis_type, className='header__analysis-type')),
                                        width='auto', className='align-self-center'),
                                ], className='justify-content-between'),

                                dbc.Row([
                                    html.Div([
                                        dcc.Tabs(id='tabs-feature-page', value='gen-metrics', children=[
                                            dcc.Tab(label='General Metrics',
                                                    value='gen-metrics',
                                                    className='header__nav-tab',
                                                    selected_className='header__nav-tab-selected',),
                                            dcc.Tab(label='Miss Predictions',
                                                    value='miss-pred',
                                                    className='header__nav-tab',
                                                    selected_className='header__nav-tab-selected',),
                                            dcc.Tab(label='Loss Clusters',
                                                    value='loss-clust',
                                                    className='header__nav-tab',
                                                    selected_className='header__nav-tab-selected'),
                                            dcc.Tab(label='xFeature Distribution',
                                                    value='xfeat-dist',
                                                    className='header__nav-tab',
                                                    selected_className='header__nav-tab-selected'),
                                            dcc.Tab(label='Similarities',
                                                    value='similarities-cf',
                                                    className='header__nav-tab',
                                                    selected_className='header__nav-tab-selected'),
                                        ], className='header__nav-row'),
                                    ]),
                                ], className='header__nav-row')
                            ], fluid=True), fluid=True, className='sticky-top'),

                        dbc.Container(html.Div(id='feature-page', children=[]), fluid=True),
                    ], fluid=True)

        return main_layout

    def run(self) -> dash.Dash:
        '''
        Spin up Rarity web application built with ``dash`` components
        '''
        app.layout = self._layout()

        @app.callback(Output('feature-page', 'children'),
                    [Input('tabs-feature-page', 'value')])
        def display_page(pathname='gen-metrics'):
            if pathname == 'gen-metrics':
                return GeneralMetrics(self.data_loader).show()
            elif pathname == 'miss-pred':
                return MissPredictions(self.data_loader).show()
            elif pathname == 'loss-clust':
                return LossClusters(self.data_loader).show()
            elif pathname == 'xfeat-dist':
                return FeatureDistribution(self.data_loader).show()
            elif pathname == 'similarities-cf':
                return SimilaritiesCF(self.data_loader).show()
            else:
                return html.Div([html.H3('feature page {}'.format(pathname))], style={'padding-left': '30px'})

        MissPredictions(self.data_loader).callbacks()
        LossClusters(self.data_loader).callbacks()
        FeatureDistribution(self.data_loader).callbacks()
        SimilaritiesCF(self.data_loader).callbacks()

        app.run_server(debug=False, port=8000)

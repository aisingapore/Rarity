import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from tenjin.features import GeneralMetrics, MissPredictions, LossClusters, FeatureDistribution, SimilaritiesCF
from tenjin.app import app


class GapAnalyzer:
    def __init__(self, data_loader, user_defined_title=None):
        self.data_loader = data_loader
        self.analysis_type = self.data_loader.get_analysis_type().replace('-', ' ').title()
        self.usr_defined_title = user_defined_title

    def _layout(self):
        main_layout = dbc.Container([
                        dbc.Jumbotron(
                            dbc.Container([
                                dbc.Row([
                                    dbc.Col(html.Div([
                                        html.Img(className='header__aisg-logo', src='assets/aisg-logo.png'),
                                        html.H4('Gap Analysis with Tenjin 3.0', className='header__tenjin-title'),
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

    def run(self):
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

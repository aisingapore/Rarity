import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from tenjin.interpreters.structured_data import IntSimilaritiesCounterFactuals
from tenjin.visualizers import shared_viz_component as viz_shared
from tenjin.utils.common_functions import is_regression, get_max_value_on_slider
from tenjin.utils import style_configs


def generate_similarities_reg(data_loader, user_defined_idx, top_n=3):
    df_viz = IntSimilaritiesCounterFactuals(data_loader).xform(user_defined_idx, top_n)
    feature_cols = list(data_loader.get_features().columns)

    category = ['User_defined_idx']
    category_top_n = [f'Top_{i + 1}' for i in range(len(df_viz) - 1)]  # len(df_viz) - 1 as first row is user_defined_idx
    category = category + category_top_n
    df_viz.insert(0, 'category', category)

    try:
        # to limit table cell having values with long decimals for better viz purpose
        df_viz[list(df_viz.columns)[3:]] = df_viz[list(df_viz.columns)[3:]].round(2)
    except TypeError:
        df_viz

    table_obj = viz_shared.reponsive_table_to_filtered_datapoints_similaritiesCF(df_viz,
                                                                                list(df_viz.columns),
                                                                                feature_cols,
                                                                                style_configs.DEFAULT_HEADER_STYLE,
                                                                                'csv')
    return table_obj


class SimilaritiesCF:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.df_features = data_loader.get_features()
        self.user_defined_idx = 1
        self.top_n = 3

        if is_regression(self.analysis_type):
            self.table_obj_reg = generate_similarities_reg(self.data_loader, self.user_defined_idx, self.top_n)

    def show(self):
        options_feature_ls = [{'label': f'{col}', 'value': f'{col}'} for col in self.df_features.columns]
        select_display_num_header = 'Select number of records to display '
        select_header_subnote = '( ranked by calculated distance on overall feature similarites referencing to the index defined above ):'
        similaritiesCF = dbc.Container([
                                    html.Div([
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Row(html.Div(html.H6('Specify data index ( default index: 1 ) :'),
                                                                        className='h6__similaritiesCF-index')),
                                                dbc.Row(dbc.Input(id='input-range-to-slice-kldiv-featdist',
                                                                    placeholder='example:  1   OR   12, 123, 1234',
                                                                    type='text',
                                                                    value=None))], width=6),
                                            dbc.Col([
                                                dbc.Row(html.Div(html.H6('Select feature to exclude from similarities calculation '
                                                                        '( if applicable ) :'),
                                                                        className='h6__feature-to-exclude')),
                                                dbc.Row(dbc.Col(dcc.Dropdown(id='select-feature-to-exclude-featdist', 
                                                                            options=options_feature_ls,
                                                                            value=[], multi=True)))], width=6)]),
                                        html.Br(),
                                        html.Br(),
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Row(html.Div(html.H6(select_display_num_header), className='h6__display-top-bottom-n')),
                                                dbc.Row(html.Div(html.Span(select_header_subnote),
                                                                className='text__display-header-similaritiesCF')),
                                                dcc.Slider(id='select-slider-top-bottom-range-featdist',
                                                    min=1,
                                                    max=get_max_value_on_slider(self.df_features, 'similaritiesCF'),  # max at 10
                                                    step=1,
                                                    value=3,
                                                    marks=style_configs.DEFAULT_SLIDER_RANGE)], width=10),
                                            dbc.Col(dbc.Row(
                                                        dcc.Loading(id='loading-output-specific-feat-featdist',
                                                            type='circle', color='#a80202'),
                                                    justify='left', className='loading__specific-feat-featdist'), width=1),
                                            dbc.Col(dbc.Row(
                                                        dbc.Button("Update",
                                                                    id='button-featdist-specific-feat-update',
                                                                    n_clicks=0,
                                                                    className='button__update-dataset'), justify='right'))])
                                    ], className='border__select-dataset'),
                                    html.Br(),
                                    html.Br(),
                                    dbc.Row(
                                        html.Div(self.table_obj_reg, className='div__table-proba-misspred'),
                                        justify='center')
                        ], fluid=True)
        return similaritiesCF

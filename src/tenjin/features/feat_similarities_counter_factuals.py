import pandas as pd

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from tenjin.app import app
from tenjin.interpreters.structured_data import IntSimilaritiesCounterFactuals
from tenjin.visualizers import shared_viz_component as viz_shared
from tenjin.utils import style_configs
from tenjin.utils.common_functions import is_regression, is_classification, get_max_value_on_slider, \
                                            detected_bimodal, detected_invalid_index_inputs


def generate_similarities(data_loader, user_defined_idx, feature_to_exclude=None, top_n=3):
    '''
    Applicable to both regression and classification
    '''
    df_viz, idx_for_top_n, calculated_distance = IntSimilaritiesCounterFactuals(data_loader).xform(user_defined_idx, feature_to_exclude, top_n)
    df_top_n = _base_df_by_calculated_distance(df_viz, idx_for_top_n, calculated_distance)

    category = ['User_defined_idx']
    category_top_n = [f'Top_{i + 1}' for i in range(len(df_top_n) - 1)]  # len(df_viz) - 1 as first row is user_defined_idx
    category = category + category_top_n
    df_top_n.insert(0, 'category', category)

    feature_cols = list(data_loader.get_features().columns)
    if is_regression(data_loader.get_analysis_type()):
        try:
            # to limit table cell having values with long decimals for better viz purpose
            df_top_n[list(df_top_n.columns)[3:]] = df_top_n[list(df_top_n.columns)[3:]].round(2)
        except TypeError:
            df_top_n
    elif is_classification(data_loader.get_analysis_type()):
        for col in df_top_n.columns:
            if 'float' in str(df_top_n[col].dtypes):
                df_top_n[col] = df_top_n[col].apply(lambda x: round(x, 4))

    table_obj = viz_shared.reponsive_table_to_filtered_datapoints_similaritiesCF(df_top_n,
                                                                                list(df_top_n.columns),
                                                                                feature_cols,
                                                                                style_configs.DEFAULT_HEADER_STYLE,
                                                                                'csv')
    return table_obj


def generate_counterfactuals(data_loader, user_defined_idx, feature_to_exclude=None, top_n=3):
    '''
    Applicable to classification only
    '''
    org_data_size = len(data_loader.get_features())
    df_viz, idx_sorted_by_distance, calculated_distance = IntSimilaritiesCounterFactuals(data_loader).xform(user_defined_idx,
                                                                                                            feature_to_exclude,
                                                                                                            org_data_size)
    df_top_n = _base_df_by_calculated_distance(df_viz, idx_sorted_by_distance, calculated_distance)

    table_objs = []
    for model in data_loader.get_model_list():
        usr_pred_label = df_top_n.loc[lambda x: x['index'] == user_defined_idx, f'yPred_{model}'].values[0]
        usr_true_label = df_top_n.loc[lambda x: x['index'] == user_defined_idx, 'yTrue'].values[0]

        df_user_idx = df_top_n.loc[lambda x: x['index'] == user_defined_idx, :]
        df_filtered_cf = df_top_n[(df_top_n['yTrue'] == usr_true_label) & (df_top_n[f'yPred_{model}'] != usr_pred_label)].head(top_n)
        df_top_n_cf = pd.concat([df_user_idx, df_filtered_cf], axis=0)

        category = ['User_defined_idx']
        category_top_n = [f'Top_CounterFactual_{i + 1}' for i in range(len(df_top_n_cf) - 1)]  # len(df_top_n_cf)-1 : 1st row is user_defined_idx
        category = category + category_top_n
        df_top_n_cf.insert(0, 'category', category)

        feature_cols = list(data_loader.get_features().columns)
        for col in df_top_n_cf.columns:
            if 'float' in str(df_top_n_cf[col].dtypes):
                df_top_n_cf[col] = df_top_n_cf[col].apply(lambda x: round(x, 4))

        table_obj = viz_shared.reponsive_table_to_filtered_datapoints_similaritiesCF(df_top_n_cf,
                                                                                    list(df_top_n_cf.columns),
                                                                                    feature_cols,
                                                                                    style_configs.DEFAULT_HEADER_STYLE,
                                                                                    'csv')
        table_objs.append(table_obj)
    return table_objs


def _base_df_by_calculated_distance(df, idx_sorted_by_distance, calculated_distance):
    df_top_n = pd.DataFrame()
    df_top_n['index'] = idx_sorted_by_distance
    df_top_n['calculated_distance'] = calculated_distance
    df_top_n = df_top_n.merge(df, how='left', on='index')
    return df_top_n


def _table_objs_similarities(data_loader, user_defined_idx, feature_to_exclude, top_n):
    table_objs_similarities = []
    for idx in str(user_defined_idx).replace(' ', '').split(','):
        table_obj_similarities = generate_similarities(data_loader, int(idx), feature_to_exclude, top_n)
        row_layout_single_idx_table_obj = dbc.Row(
                                                html.Div(table_obj_similarities,
                                                        className='div__table-proba-misspred'),
                                                justify='center')
        table_objs_similarities.append(row_layout_single_idx_table_obj)
    return table_objs_similarities


def _table_objs_counterfactuals(data_loader, user_defined_idx, feature_to_exclude, top_n):
    models = data_loader.get_model_list()
    table_objs_counterfactuals = []
    for idx in str(user_defined_idx).replace(' ', '').split(','):
        table_objs_cf_single_idx = generate_counterfactuals(data_loader, int(idx), feature_to_exclude, top_n)
        table_objs_counterfactuals.append(table_objs_cf_single_idx)

    viz_table_objs_counterfactuals = []
    for table_objs_cf in table_objs_counterfactuals:
        if detected_bimodal(models):
            cf_table_obj_single_idx = [dbc.Row([html.H5('Counter-Factuals for model : ',
                                                        id='title-after-topn-bottomn-reg',
                                                        className='h5__counterfactuals-section-title'),
                                                        html.H5(f'{models[0]}', className='h5__counterfactual-model')]),
                                                dbc.Row(
                                                    html.Div(table_objs_cf[0],
                                                            id='table-obj-similarities-bm-1',
                                                            className='div__table-proba-misspred'),
                                                    justify='center'),
                                                html.Br(),
                                                dbc.Row([dbc.Row(html.H5('Counter-Factuals for model : ',
                                                                id='title-after-topn-bottomn-reg',
                                                                className='h5__counterfactuals-section-title')),
                                                        dbc.Row(html.H5(f'{models[1]}', className='h5__counterfactual-model'))]),
                                                dbc.Row(
                                                    html.Div(table_objs_cf[1],
                                                            id='table-obj-similarities-bm-2',
                                                            className='div__table-proba-misspred'),
                                                    justify='center')]
            viz_table_objs_counterfactuals += cf_table_obj_single_idx
        else:
            cf_table_obj_single_idx = [dbc.Row([html.H5('Counter-Factuals for model : ',
                                                        id='title-after-topn-bottomn-reg',
                                                        className='h5__counterfactuals-section-title'),
                                                        html.H5(f'{models[0]}', className='h5__counterfactual-model')]),
                                                dbc.Row(
                                                    html.Div(table_objs_cf[0],
                                                            id='table-obj-similarities-sm',
                                                            className='div__table-proba-misspred'),
                                                    justify='center')]
            viz_table_objs_counterfactuals += cf_table_obj_single_idx
    return viz_table_objs_counterfactuals


class SimilaritiesCF:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.df_features = data_loader.get_features()
        self.feature_to_exclude = []
        self.user_defined_idx = '1'
        self.top_n = 3
        self.table_objs_similarities = _table_objs_similarities(self.data_loader, self.user_defined_idx, self.feature_to_exclude, self.top_n)

    def show(self):
        options_feature_ls = [{'label': f'{col}', 'value': f'{col}'} for col in self.df_features.columns]
        shared_layout_reg_cls = [
                                    html.Div([
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Row(html.Div(html.H6('Specify data index ( default index: 1 ) :'),
                                                                        className='h6__similaritiesCF-index')),
                                                dbc.Row(dbc.Input(id='input-specific-idx-similaritiesCF',
                                                                    placeholder='example:  1   OR   12, 123, 1234',
                                                                    type='text',
                                                                    value=None)),
                                                dbc.Row(html.Div(html.Pre(style_configs.input_range_subnote(self.df_features),
                                                                            className='text__range-header-kldiv-featfist')))], width=6),
                                            dbc.Col([
                                                dbc.Row(html.Div(html.H6('Select feature to exclude from similarities calculation '
                                                                        '( if applicable ) :'),
                                                                        className='h6__feature-to-exclude')),
                                                dbc.Row(dbc.Col(dcc.Dropdown(id='select-feature-to-exclude-similaritiesCF', 
                                                                            options=options_feature_ls,
                                                                            value=[], multi=True)))], width=6)]),
                                        html.Br(),
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Row(html.Div(html.H6('Select number of records to display'),
                                                                            className='h6__display-top-bottom-n')),
                                                dbc.Row(html.Div(html.Span('( ranked by calculated distance on overall feature similarites '
                                                                            'referencing to the index defined above ):'),
                                                                className='text__display-header-similaritiesCF')),
                                                dcc.Slider(id='select-slider-top-n-similaritiesCF',
                                                    min=1,
                                                    max=get_max_value_on_slider(self.df_features, 'similaritiesCF'),  # max at 10
                                                    step=1,
                                                    value=3,
                                                    marks=style_configs.DEFAULT_SLIDER_RANGE)], width=10),
                                            dbc.Col(dbc.Row(
                                                        dcc.Loading(id='loading-output-similaritiesCF',
                                                            type='circle', color='#a80202'),
                                                    justify='left', className='loading__similaritiesCF'), width=1),
                                            dbc.Col(dbc.Row(
                                                        dbc.Button("Update",
                                                                    id='button-similaritiesCF-update',
                                                                    n_clicks=0,
                                                                    className='button__update-dataset'), justify='right'))])
                                    ], className='border__select-dataset'),
                                    html.Div(id='alert-index-input-error-similaritiesCF'),
                                    html.Br(),
                                    dbc.Row(html.H5('Comparison based on Feature Similarities',
                                                    id='title-after-topn-bottomn-reg',
                                                    className='h5__counterfactuals-section-title'))
                                ] + [html.Div(self.table_objs_similarities, id='table-objs-similarities-shared')]

        if is_regression(self.analysis_type):
            similaritiesCF = dbc.Container(shared_layout_reg_cls + [html.Div(id='table-objs-counter-factuals')], fluid=True)

        elif is_classification(self.analysis_type):
            self.models = self.data_loader.get_model_list()
            self.table_objs_counterfactuals = _table_objs_counterfactuals(self.data_loader,
                                                                            self.user_defined_idx,
                                                                            self.feature_to_exclude,
                                                                            self.top_n)

            combined_layouts_similaritiesCF = shared_layout_reg_cls + [html.Br(),
                                                                        html.Div(self.table_objs_counterfactuals,
                                                                                id='table-objs-counter-factuals')]
            similaritiesCF = dbc.Container(combined_layouts_similaritiesCF, fluid=True)
        return similaritiesCF

    def callbacks(self):
        # callback on params related to top-n, botton-n / both on regression and classification tasks
        @app.callback(
            Output('loading-output-similaritiesCF', 'children'),
            Output('alert-index-input-error-similaritiesCF', 'children'),
            Output('table-objs-similarities-shared', 'children'),
            Output('table-objs-counter-factuals', 'children'),
            Input('button-similaritiesCF-update', 'n_clicks'),
            State('input-specific-idx-similaritiesCF', 'value'),
            State('select-feature-to-exclude-similaritiesCF', 'value'),
            State('select-slider-top-n-similaritiesCF', 'value'))
        def generate_table_objs_based_on_user_selected_params(click_count, specific_idx, feature_to_exclude, top_n):
            if click_count > 0:
                if specific_idx is None:  # during first spin-up
                    specific_idx = self.user_defined_idx  # default value
                idx_input_err_alert = style_configs.no_error_alert()

                if detected_invalid_index_inputs(specific_idx, self.df_features):
                    idx_input_err_alert = style_configs.activate_invalid_index_input_alert(self.df_features)
                    return '', idx_input_err_alert, dash.no_update, dash.no_update

                if is_regression(self.analysis_type):
                    table_objs_similarities = _table_objs_similarities(self.data_loader, specific_idx, feature_to_exclude, top_n)
                    return '', idx_input_err_alert, table_objs_similarities, ''
                elif is_classification(self.analysis_type):
                    table_objs_similarities = _table_objs_similarities(self.data_loader, specific_idx, feature_to_exclude, top_n)
                    table_objs_counterfactuals = _table_objs_counterfactuals(self.data_loader, specific_idx, feature_to_exclude, top_n)
                    return '', idx_input_err_alert, table_objs_similarities, table_objs_counterfactuals
            else:
                raise PreventUpdate

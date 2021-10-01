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

from typing import Union, List
import pandas as pd

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from rarity.app import app
from rarity.data_loader import CSVDataLoader, DataframeLoader
from rarity.interpreters.structured_data import IntSimilaritiesCounterFactuals
from rarity.visualizers import shared_viz_component as viz_shared
from rarity.utils import style_configs
from rarity.utils.common_functions import is_regression, is_classification, get_max_value_on_slider, \
                                            detected_bimodal, detected_invalid_index_inputs


def generate_similarities(data_loader: Union[CSVDataLoader, DataframeLoader], user_defined_idx, feature_to_exclude=None, top_n=3):
    '''
    Tapout table collating feature info corresponding to user defined index and top N index based on distance score.
    Applicable to both regression and classification

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module
        user_defined_idx (int):
            Index of the data point of interest specified by user
        feature_to_exclude (:obj:`List[str]`, `optional`):
                A list of features to be excluded from the ranking and similarities distance calculation
        top_n (int):
                Number indicating the max limit of records to be displayed based on the distance ranking

    Returns:
        :obj:`~dash_table.DataTable`:
            table object outlining the dataframe content with dynamic-conditional styles
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


def generate_counterfactuals(data_loader: Union[CSVDataLoader, DataframeLoader], user_defined_idx, feature_to_exclude=None, top_n=3):
    '''
    Tapout table collating feature info corresponding to user defined index and top N index based on distance score with condition
    that the prediction labels of top N index differ from prediction label of user defined index
    Applicable to both classification only

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module
        user_defined_idx (int):
            Index of the data point of interest specified by user
        feature_to_exclude (:obj:`List[str]`, `optional`):
                A list of features to be excluded from the ranking and similarities distance calculation
        top_n (int):
                Number indicating the max limit of records to be displayed based on the distance ranking

    Returns:
        :obj:`~dash_table.DataTable`:
            table object outlining the dataframe content with dynamic-conditional styles
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


def _base_df_by_calculated_distance(df: pd.DataFrame, idx_sorted_by_distance: List[int], calculated_distance: List[float]):
    '''
    Setup new dataframe storing calculated distance info
    '''
    df_top_n = pd.DataFrame()
    df_top_n['index'] = idx_sorted_by_distance
    df_top_n['calculated_distance'] = calculated_distance
    df_top_n = df_top_n.merge(df, how='left', on='index')
    return df_top_n


def _table_objs_similarities(data_loader: Union[CSVDataLoader, DataframeLoader],
                            user_defined_idx: int,
                            feature_to_exclude: List[str],
                            top_n: int):
    '''
    List collating layouts for similarities table based on user index/indices
    '''
    table_objs_similarities = []
    for idx in str(user_defined_idx).replace(' ', '').split(','):
        table_obj_similarities = generate_similarities(data_loader, int(idx), feature_to_exclude, top_n)
        row_layout_single_idx_table_obj = dbc.Row(
                                                html.Div(table_obj_similarities,
                                                        className='div__table-proba-misspred'),
                                                justify='center')
        table_objs_similarities.append(row_layout_single_idx_table_obj)
    return table_objs_similarities


def _table_objs_counterfactuals(data_loader: Union[CSVDataLoader, DataframeLoader],
                                user_defined_idx: int,
                                feature_to_exclude: List[str],
                                top_n: int):
    '''
    List collating layouts for counterfactual tables based on user index/indices
    '''
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
    '''
    Main integration for feature component on Similarities-CounterFactuals

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Important Attributes:

        analysis_type (str):
            Analysis type defined by user during initial inputs preparation via data_loader stage.
        df_features (:obj:`~pandas.DataFrame`):
            Dataframe storing all features used in dataset
        feature_to_exclude (:obj:`List[str]`, `optional`):
            A list of features to be excluded from the ranking and similarities distance calculation
        user_defined_idx (int):
            Index of the data point of interest specified by user
        top_n (int):
                Number indicating the max limit of records to be displayed based on the distance ranking

    Returns:
        :obj:`~dash_core_components.Container`:
            styled dash components displaying graph and/or table objects
    '''
    def __init__(self, data_loader: Union[CSVDataLoader, DataframeLoader]):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.df_features = data_loader.get_features()
        self.feature_to_exclude = []
        self.user_defined_idx = '1'
        self.top_n = 3
        self.table_objs_similarities = _table_objs_similarities(self.data_loader, self.user_defined_idx, self.feature_to_exclude, self.top_n)

    def show(self):
        '''
        Method to tapout styled html for Similarities-CounterFactuals
        '''
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
            '''
            Callbacks functionalities on params related to top-n, botton-n / both on regression and classification tasks
            '''
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

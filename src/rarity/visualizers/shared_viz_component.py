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

from typing import List, Dict
import dash_table


def reponsive_table_to_filtered_datapoints(data: Dict, customized_cols: List[str], header: Dict, exp_format: str):
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
    table_obj = dash_table.DataTable(
                    data=data,
                    columns=[{'id': c, 'name': c} for c in customized_cols],
                    page_action='none',
                    fixed_rows={'headers': True},
                    fixed_columns={'headers': True, 'data': 1},
                    style_header=header,
                    style_data={'whiteSpace': 'normal', 'height': 'auto'},
                    style_cell={'textAlign': 'center',
                                'border': '1px solid rgb(229, 211, 197)',
                                'font-family': 'Arial',
                                'margin-bottom': '0',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'minWidth': '200px',
                                'width': '200px',
                                'maxWidth': '200px'},
                    style_table={'width': 1400,
                                'height': 200,
                                'margin': 'auto'},
                    export_format=exp_format),
    return table_obj


def reponsive_table_to_filtered_datapoints_similaritiesCF(df, customized_cols, feature_cols, header, exp_format):
    '''
    Create table outlining dataframe content specific to Counter-Factuals component

    Arguments:
        df (:obj:`~pd.DataFrame`):
            dataframe containing calculated distance info
        customized_cols (:obj:`List[str]`):
            list of customized column names
        feature_cols (:obj:`List[str]`):
            list of feature column names
        header (:obj:`Dict`):
            dictionary format storing the style info for table header
        exp_format (str):
            text info indicating the export format

    Returns:
        :obj:`~dash_table.DataTable`:
            table object outlining the dataframe content with dynamic-conditional styles
    '''
    data = df.to_dict('records')
    table_obj = dash_table.DataTable(
                    data=data,
                    columns=[{'id': c, 'name': c} for c in customized_cols],
                    page_action='none',
                    fixed_rows={'headers': True},
                    fixed_columns={'headers': True, 'data': 1},
                    style_header=header,
                    style_data={'whiteSpace': 'normal', 'height': 'auto'},
                    style_cell={'textAlign': 'center',
                                'border': '1px solid rgb(229, 211, 197)',
                                'font-family': 'Arial',
                                'margin-bottom': '0',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'minWidth': '200px',
                                'width': '200px',
                                'maxWidth': '200px'},
                    style_table={'width': 1400,
                                'height': 200,
                                'margin': 'auto'},
                    style_header_conditional=[
                        {
                            'if': {'column_id': col},
                            'backgroundColor': '#b69e8d',
                            'border': '1px solid white',
                        } for col in feature_cols
                    ],
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'category'},
                            'border-top': '1px solid rgb(229, 211, 197)',
                            'fontWeight': 'bold'}
                    ] + [
                        {
                            'if': {'filter_query': '{category} = "User_defined_idx"'},
                            'backgroundColor': 'rgb(229, 240, 247)',
                        }
                    ],
                    export_format=exp_format),
    return table_obj

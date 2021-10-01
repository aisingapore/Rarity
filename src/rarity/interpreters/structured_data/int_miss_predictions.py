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

from typing import Union
import pandas as pd

from rarity.data_loader import CSVDataLoader, DataframeLoader
from rarity.interpreters.structured_data.base_interpreters import BaseInterpreters
from rarity.utils.common_functions import is_regression, is_classification


class IntMissPredictions(BaseInterpreters):
    '''
    Transform raw data into input format suitable for visualization on miss-prediction points

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Returns:
        :obj:`~pd.DataFrame`:
            Dataframe with essential info suitable for visualization on regression task

    .. note::

        if classification, returns:

            Compact outputs consist of the followings

            - ls_dfs_viz (:obj:`List[~pd.DataFrame]`): list of dataframes for overview visualization need
            - ls_class_labels (:obj:`List[str]`): list of class labels
            - ls_dfs_by_label  (:obj:`List[~pd.DataFrame]`): list of dataframes by individual label class
            - ls_dfs_by_label_state (:obj:`List[~pd.DataFrame]`): list of dataframes storing basic stats of each label class
    '''
    def __init__(self, data_loader: Union[CSVDataLoader, DataframeLoader]):
        super().__init__(data_loader)

    def xform(self):
        '''
        Core transformation function to tap-out data into input format suitable for plotly graph
        '''
        if is_regression(self.analysis_type):
            df = super().get_df_with_offset_values()
            return df

        elif is_classification(self.analysis_type):
            ls_dfs_viz, ls_class_labels = super().get_df_with_probability_values()

            ls_dfs_by_label = []
            ls_dfs_by_label_state = []
            for df_viz in ls_dfs_viz:
                # tapout df that is specific to each label-class
                dfs_specific_label = []
                dfs_state = []
                for label in ls_class_labels:
                    df_label = df_viz[df_viz['yTrue'] == int(label)]
                    df_label = df_label[['yTrue', label, 'model', 'yPred-label', 'pred_state']]
                    dfs_specific_label.append(df_label)

                    is_exist_correct = True if len(df_label[df_label['pred_state'] == 'correct']) != 0 else False
                    is_exist_misspred = True if len(df_label[df_label['pred_state'] == 'miss-predict']) != 0 else False
                    state_dict = {}
                    state_dict['sample_size: '] = len(df_label)
                    state_dict['correct: '] = len(df_label[df_label['pred_state'] == 'correct']) if is_exist_correct else 0
                    state_dict['miss-predict:'] = len(df_label[df_label['pred_state'] == 'miss-predict']) if is_exist_misspred else 0
                    state_dict['accuracy: '] = round(state_dict['correct: '] / len(df_label), 4)
                    df_state = pd.DataFrame(state_dict, index=[0],).transpose().reset_index().rename(columns={0: 'state_value'})
                    dfs_state.append(df_state)

                ls_dfs_by_label.append(dfs_specific_label)
                ls_dfs_by_label_state.append(dfs_state)
            return ls_dfs_viz, ls_class_labels, ls_dfs_by_label, ls_dfs_by_label_state

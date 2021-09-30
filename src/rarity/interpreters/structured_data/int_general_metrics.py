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
import numpy as np
from sklearn import metrics

from rarity.data_loader import CSVDataLoader, DataframeLoader
from rarity.utils.common_functions import is_regression, is_classification

ERR_DESC = ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error', 'R Squared']
ERR_NAMES = ['MAE', 'MSE', 'RMSE', 'R2']
METRICS_FORMULA = ['![MAE](assets/MAE.png)', '![MSE](assets/MSE.png)', 
                    '![RMSE](assets/RMSE.png)', '![R2](assets/R2.png)']
ERR_METRICS_DICT = {}
ERR_METRICS_DICT['Metrics Description'] = ERR_DESC
ERR_METRICS_DICT['Formula'] = METRICS_FORMULA
ERR_METRICS_DICT['Metrics Name'] = ERR_NAMES


class IntGeneralMetrics:
    """
    Transform raw data into input format suitable for visualization. General metrics cover
    confusion matrix, classification report, roc curve and precisionRecall curve

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

        viz_plot (str):
            Supported visualization types : ``confMat``, ``classRpt``, ``rocAUC``, ``preRacall``, ``stdErr``, None

    Important Attributes:

        - analysis_type (str):
            Analysis type defined by user during initial inputs preparation via data_loader stage.

    Returns:
        :obj:`~pd.DataFrame`:
            Dataframe with essential info suitable for visualization on regression task

    .. note::

            if classification, returns:

            - yTrue data in :obj:`~pd.Series`
            - yPred data in :obj:`~pd.Series` for [``confMat``, ``classRpt``] or :obj:`~pd.Dataframe` for [``rocAuc``, ``precRecall``]

            If multiclass, returns:

            - yPred data in :obj:`List[List[Tuple]]` pairing class label and yPred in :obj:`~pd.Series`
            - model_names in :obj:`List[str]`
    """
    def __init__(self, data_loader: Union[CSVDataLoader, DataframeLoader], viz_plot: str = None):
        self.data_loader = data_loader
        self.viz_plot = viz_plot
        self.analysis_type = self.data_loader.get_analysis_type()

    def _std_err_metrics(self, yTrue: pd.Series, yPred: pd.Series) -> List:
        mae = metrics.mean_absolute_error(yTrue, yPred)
        mse = metrics.mean_squared_error(yTrue, yPred)
        rmse = np.sqrt(metrics.mean_squared_error(yTrue, yPred))
        r2_score = metrics.r2_score(yTrue, yPred)
        return [int(mae), int(mse), int(rmse), round(r2_score, 4)]

    def xform(self):
        '''
        Core transformation function to tap-out data into input format suitable for plotly graph
        '''
        if is_regression(self.analysis_type):
            if self.viz_plot == 'stdErr':
                yTrue = self.data_loader.get_yTrue()
                yPreds = self.data_loader.get_yPreds()
                models = self.data_loader.get_model_list()
                if len(models) == 1:
                    ERR_METRICS_DICT[f'Model_{models[0]}'] = self._std_err_metrics(yTrue, yPreds)
                elif len(models) > 1:
                    for i in range(len(models)):
                        ERR_METRICS_DICT[f'Model_{models[i]}'] = self._std_err_metrics(yTrue, yPreds[yPreds.columns[i]])
                df = pd.DataFrame(ERR_METRICS_DICT)
            else:
                df = self.data_loader.get_all()
            return df

        elif is_classification(self.analysis_type):
            model_names = self.data_loader.get_model_list()
            yTrue = self.data_loader.get_yTrue()
            yTrue = yTrue['yTrue'].astype('string')
            preds = self.data_loader.get_yPreds()

            if self.viz_plot in ['confMat', 'classRpt']:
                yPred = [pred['yPred-label'] for pred in preds]
            elif self.viz_plot in ['rocAuc', 'precRecall']:
                yPred = [pred[pred.columns[-3]] for pred in preds]
                is_multiclass = len(set(yTrue)) > 2
                if is_multiclass:
                    yPred = []
                    for pred in preds:
                        label_keys = pred['yPred-label']
                        pred_values = pred[pred.columns[:-2]].max(axis=1)
                        pred_tmp = [(label_keys[i], pred_values[i]) for i in range(len(label_keys))]
                        yPred.append(pred_tmp)
            return yTrue, yPred, model_names

"""
Version : 
---------
draft 3.0 [ 5 July 2021 - dash ]

"""
import pandas as pd
import numpy as np
from sklearn import metrics


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
    - Transform raw data into input format suitable for plotting with plotly
    - General metrics cover confusion matrix, classification report, roc curve and precisionRecall curve

    Arguments:
        data_loader {class object} -- class object from data_loader pipeline
        viz_plot {str} -- visualization type : 'confMat', 'classRpt', 'rocAUC', 'preRacall', 'stdErr', None

    Returns:
        either (if analysis_type==regression)
            yTrue {pd.series} -- actual labels
            yPred {list of list} -- Predicted labels of different models
            model_names {list} -- name of models used to produce yPred
        or (if analysis_type==classification)
            df {dataframe} -- dataframe used to plot fig-obj in visualizer module
        """
    def __init__(self, data_loader, viz_plot=None):
        self.data_loader = data_loader
        self.viz_plot = viz_plot
        self.analysis_type = self.data_loader.get_analysis_type()

    def _std_err_metrics(self, yTrue, yPred):
        mae = metrics.mean_absolute_error(yTrue, yPred)
        mse = metrics.mean_squared_error(yTrue, yPred)
        rmse = np.sqrt(metrics.mean_squared_error(yTrue, yPred))
        r2_score = metrics.r2_score(yTrue, yPred)
        return [int(mae), int(mse), int(rmse), round(r2_score, 4)]

    def xform(self):
        if self.analysis_type == 'regression':
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

        elif 'classification' in self.analysis_type:
            model_names = self.data_loader.get_model_list()
            yTrue = self.data_loader.get_yTrue()
            yTrue = yTrue['yTrue'].astype('string')
            preds = self.data_loader.get_yPreds()

            if self.viz_plot in ['confMat', 'classRpt']:
                yPred = [pred['yPred-label'] for pred in preds]
            elif self.viz_plot in ['rocAuc', 'precRecall']:
                is_multiclass = len(set(yTrue)) > 2
                if is_multiclass:
                    yPred = []
                    for pred in preds:
                        label_keys = pred['yPred-label']
                        pred_values = pred[pred.columns[:-2]].max(axis=1)
                        pred_tmp = [(label_keys[i], pred_values[i]) for i in range(len(label_keys))]
                        yPred.append(pred_tmp)
                else:
                    yPred = [pred[pred.columns[-3]] for pred in preds]
            return yTrue, yPred, model_names

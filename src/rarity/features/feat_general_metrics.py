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

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from rarity.data_loader import CSVDataLoader, DataframeLoader
from rarity.interpreters.structured_data import IntGeneralMetrics
from rarity.visualizers import general_metrics as viz_general
from rarity.utils.common_functions import is_regression, is_classification


def fig_confusion_matrix(data_loader: Union[CSVDataLoader, DataframeLoader]):
    """
    Create confusion matrix

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Returns:
        :obj:`~plotly.graph_objects.Figure`:
            figure displaying confusion matrix details
    """
    yTrue, yPred, model_names = IntGeneralMetrics(data_loader, 'confMat').xform()
    fig_objs = viz_general.plot_confusion_matrix(yTrue, yPred, model_names)
    return fig_objs


def fig_classification_report(data_loader: Union[CSVDataLoader, DataframeLoader]):
    """
    Create classification report in table form

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Returns:
        :obj:`List[~plotly.graph_objects.Figure]`:
            list of tables displaying classification report details
    """
    yTrue, yPred, model_names = IntGeneralMetrics(data_loader, 'classRpt').xform()
    fig_objs = viz_general.plot_classification_report(yTrue, yPred, model_names)
    return fig_objs


def fig_roc_curve(data_loader: Union[CSVDataLoader, DataframeLoader]):
    """
    Display roc curve for comparison on various models

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Returns:
        :obj:`~plotly.graph_objects.Figure`:
            figure displaying line curves comparing roc-auc score for various models
    """
    yTrue, yPred, model_names = IntGeneralMetrics(data_loader, 'rocAuc').xform()
    fig_obj = viz_general.plot_roc_curve(yTrue, yPred, model_names)
    return fig_obj


def fig_precisionRecall_curve(data_loader: Union[CSVDataLoader, DataframeLoader]):
    """
    Display precision-recall curve for comparison on various models

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Returns:
        :obj:`~plotly.graph_objects.Figure`:
            figure displaying line curves comparing precision-recall for various models
    """
    yTrue, yPred, model_names = IntGeneralMetrics(data_loader, 'precRecall').xform()
    fig_obj = viz_general.plot_precisionRecall_curve(yTrue, yPred, model_names)
    return fig_obj


def fig_prediction_actual_comparison(data_loader: Union[CSVDataLoader, DataframeLoader]):
    """
    Display scatter plot for comparison on actual values (yTrue) vs prediction values (yPred)

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Returns:
        :obj:`~plotly.graph_objects.Figure`:
            figure displaying scatter plot comparing actual values vs prediction values
    """
    df = IntGeneralMetrics(data_loader).xform()
    fig_obj = viz_general.plot_prediction_vs_actual(df)
    return fig_obj


def fig_prediction_offset_overview(data_loader: Union[CSVDataLoader, DataframeLoader]):
    """
    Display scatter plot for overview on prediction offset values

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Returns:
        :obj:`~plotly.graph_objects.Figure`:
            figure displaying scatter plot outlining overview on prediction offset values
    """
    df = IntGeneralMetrics(data_loader).xform()
    fig_obj = viz_general.plot_prediction_offset_overview(df)
    return fig_obj


def fig_standard_error_metrics(data_loader: Union[CSVDataLoader, DataframeLoader]):
    """
    Display table comparing various standard metrics for regression task

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Returns:
        :obj:`~dash_table.DataTable`:
            table object comparing various standard metrics for regression task
    """
    df = IntGeneralMetrics(data_loader, 'stdErr').xform()
    fig_obj = viz_general.plot_std_error_metrics(df)
    return fig_obj


class GeneralMetrics:
    '''
    Main integration for feature component on General Metrics.

        - On Regression: ``Prediction vs Actual``, ``Prediction vs Offset``
        - On Classification: ``Confusion Matrix``, ``Classification Report``, ``ROC``, ``Precision-Recall``

    Arguments:
        data_loader (:class:`~rarity.data_loader.CSVDataLoader` or :class:`~rarity.data_loader.DataframeLoader`):
            Class object from data_loader module

    Returns:
        :obj:`~dash_core_components.Container`:
            styled dash components displaying graph and/or table objects
    '''
    def __init__(self, data_loader: Union[CSVDataLoader, DataframeLoader]):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()

    def show(self):
        if is_regression(self.analysis_type):
            self.pred_actual = fig_prediction_actual_comparison(self.data_loader)
            self.pred_offset = fig_prediction_offset_overview(self.data_loader)
            self.std_error_metrics = fig_standard_error_metrics(self.data_loader)
            gen_metrics = dbc.Container([
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(figure=self.pred_actual)
                                    ], className="border__common-gen-metrics-reg-left"),
                                    dbc.Col([
                                        dcc.Graph(figure=self.pred_offset)
                                    ], className="border__common-gen-metrics-reg-right"),
                                ]), 

                                html.Div(html.Div(self.std_error_metrics, className="div__std-err")),
                                html.Br(),
                        ], fluid=True)

        elif is_classification(self.analysis_type):
            self.conf_matrix = fig_confusion_matrix(self.data_loader)
            self.cls_report = fig_classification_report(self.data_loader)
            self.roc = fig_roc_curve(self.data_loader)
            self.prec_recall = fig_precisionRecall_curve(self.data_loader)
            if len(self.conf_matrix) > 1:
                gen_metrics = dbc.Container([
                                    dbc.Row([
                                        dbc.Col([
                                            dcc.Graph(figure=self.conf_matrix[0]),
                                            dcc.Graph(figure=self.cls_report[0]),
                                        ], className="border__common-left"),

                                        dbc.Col([
                                            dcc.Graph(figure=self.conf_matrix[1]),
                                            dcc.Graph(figure=self.cls_report[1]),
                                        ], className="border__common-right")
                                    ]), 

                                    html.Div(html.Div(dcc.Graph(figure=self.roc), 
                                                                className="fig__roc-prec-recall"), 
                                                                className="border__common"),
                                    html.Div(html.Div(dcc.Graph(figure=self.prec_recall), 
                                                                className="fig__roc-prec-recall"), 
                                                                className="border__common")
                            ], fluid=True)

            elif len(self.conf_matrix) == 1:
                gen_metrics = dbc.Container([
                                    html.Div(dbc.Row([
                                        dbc.Col(dcc.Graph(figure=self.conf_matrix[0]), className="border__common-left"),
                                        dbc.Col(dcc.Graph(figure=self.cls_report[0]), className="border__common-right")
                                    ]), className="boundary__common"), 

                                    html.Div(html.Div(dcc.Graph(figure=self.roc), 
                                                                className="fig__roc-prec-recall"), 
                                                                className="border__common"),
                                    html.Div(html.Div(dcc.Graph(figure=self.prec_recall), 
                                                                className="fig__roc-prec-recall"), 
                                                                className="border__common")
                            ], fluid=True, className="boundary__common")

        return gen_metrics

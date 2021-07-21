import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from ..interpreters.structured_data import IntGeneralMetrics
from ..visualizer import general_metrics as viz_general


def fig_confusion_matrix(data_loader):
    """
    Create confusion matrix

    Arguments:
        data_loader {class object} 
        -- output from data loader pipeline 

    Returns:
        plotly graph 
        -- displaying confusion matrix details
    """
    yTrue, yPred, model_names = IntGeneralMetrics(data_loader, 'confMat').xform()
    fig_objs = viz_general.plot_confusion_matrix(yTrue, yPred, model_names)
    return fig_objs


def fig_classification_report(data_loader):
    """
    Create classification report in table form

    Arguments:
        data_loader {class object} 
        -- output from data loader pipeline 

    Returns:
        plotly table 
        -- containing classification report details
    """
    yTrue, yPred, model_names = IntGeneralMetrics(data_loader, 'classRpt').xform()
    fig_objs = viz_general.plot_classification_report(yTrue, yPred, model_names)
    return fig_objs


def fig_roc_curve(data_loader):
    """
    Display roc curve for comparison on various models

    Arguments:
        data_loader {class object} 
        -- output from data loader pipeline 

    Returns:
        plotly line curve 
        -- comparing roc-auc score for various models
    """
    yTrue, yPred, model_names = IntGeneralMetrics(data_loader, 'rocAuc').xform()
    fig_obj = viz_general.plot_roc_curve(yTrue, yPred, model_names)
    return fig_obj


def fig_precisionRecall_curve(data_loader):
    """
    Display precision-recall curve for comparison on various models

    Arguments:
        data_loader {class object} 
        -- output from data loader pipeline 

    Returns:
        plotly line curve 
        -- comparing roc-auc score for various models
    """
    yTrue, yPred, model_names = IntGeneralMetrics(data_loader, 'precRecall').xform()
    fig_obj = viz_general.plot_precisionRecall_curve(yTrue, yPred, model_names)
    return fig_obj


def fig_prediction_actual_comparison(data_loader):
    """
    Display comparison on prediction (yPred) vs actual (yTrue)

    Arguments:
        data_loader {class object} 
        -- output from data loader pipeline 

    Returns:
        plotly scatter plot
        -- comparing state of Prediction vs Actual for various models
    """
    df = IntGeneralMetrics(data_loader).xform()
    fig_obj = viz_general.plot_prediction_vs_actual(df)
    return fig_obj


def fig_prediction_offset_overview(data_loader):
    """
    Display overview of prediction offset

    Arguments:
        data_loader {class object} 
        -- output from data loader pipeline 

    Returns:
        plotly scatter plot with baseline
        -- comparing state of Prediction vs Residual / Offset for various models
    """
    df = IntGeneralMetrics(data_loader).xform()
    fig_obj = viz_general.plot_prediction_offset_overview(df)
    return fig_obj


def fig_standard_error_metrics(data_loader):
    """
    Display table comparing various standard metrics for regression task

    Arguments:
        data_loader {class object} 
        -- output from data loader pipeline 

    Returns:
        dash table
        -- comparing general metrics covering MAE, MSE, RSME, R2 for single model and bimodal
    """
    df = IntGeneralMetrics(data_loader, 'stdErr').xform()
    fig_obj = viz_general.plot_std_error_metrics(df)
    return fig_obj


class GeneralMetrics:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()

        if self.analysis_type == 'regression':
            self.pred_actual = fig_prediction_actual_comparison(self.data_loader)
            self.pred_offset = fig_prediction_offset_overview(self.data_loader)
            self.std_error_metrics = fig_standard_error_metrics(self.data_loader)
        elif 'classification' in self.analysis_type:
            self.conf_matrix = fig_confusion_matrix(self.data_loader)
            self.cls_report = fig_classification_report(self.data_loader)
            self.roc = fig_roc_curve(self.data_loader)
            self.prec_recall = fig_precisionRecall_curve(self.data_loader)

    def show(self):
        if self.analysis_type == 'regression':
            gen_metrics = dbc.Container([
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(figure=self.pred_actual),
                                    ], className="border__common"), 

                                    dbc.Col([
                                        dcc.Graph(figure=self.pred_offset),
                                    ], className="border__common")
                                ]), 

                                html.Div(html.Div(self.std_error_metrics, className="div__std-err")),
                                html.Br(),
                        ], fluid=True)

        elif 'classification' in self.analysis_type:
            if len(self.conf_matrix) > 1:
                gen_metrics = dbc.Container([
                                    dbc.Row([
                                        dbc.Col([
                                            dcc.Graph(figure=self.conf_matrix[0]),
                                            dcc.Graph(figure=self.cls_report[0]),
                                        ], className="border__common"), 

                                        dbc.Col([
                                            dcc.Graph(figure=self.conf_matrix[1]),
                                            dcc.Graph(figure=self.cls_report[1]),
                                        ], className="border__common")
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
                                        dbc.Col(dcc.Graph(figure=self.conf_matrix[0]), className="border__common"), 
                                        dbc.Col(dcc.Graph(figure=self.cls_report[0]), className="border__common")
                                    ]), className="boundary__common"), 

                                    html.Div(html.Div(dcc.Graph(figure=self.roc), 
                                                                className="fig__roc-prec-recall"), 
                                                                className="border__common"),
                                    html.Div(html.Div(dcc.Graph(figure=self.prec_recall), 
                                                                className="fig__roc-prec-recall"), 
                                                                className="border__common")
                            ], fluid=True, className="boundary__common")

        return gen_metrics

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

from typing import List
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve

import plotly.express as px
import plotly.graph_objects as go
import dash_table


def plot_confusion_matrix(yTrue: pd.Series, yPred: pd.Series, model_names: List):
    """
    Create confusion matrix

    Arguments:
        yTrue (:obj:`pd.Series`):
            true labels, output from int_general_metrics
        yPred (:obj:`pd.Series`):
            predicted labels, output from int_general_metrics
        model_names (:obj:`List[str]`):
            model names, output from interpreter int_general_metrics

    Returns:
        :obj:`~plotly.graph_objects.Figure`:
            figure displaying confusion matrix details
    """
    fig_objs = []
    for i in range(len(yPred)):
        conf_matrix = confusion_matrix(yTrue, yPred[i], labels=list(sorted(set(yTrue))))

        fig = px.imshow(conf_matrix, 
                        labels=dict(x="Predicted Label", y="True Label", color="No.of Label"),
                        color_continuous_scale=px.colors.sequential.Viridis,
                        x=list(sorted(set(yPred[i]))),  # x = y
                        y=list(sorted(set(yTrue))),  # y = x
                        title=f'Confusion Matrix : <b>{model_names[i]}</b>') 
        fig.update_layout(title={'y': 0.90, 
                                 'x': 0.5, 
                                 'xanchor': 'center', 
                                 'yanchor': 'top', 
                                 },
                          margin=dict(r=180),
                          xaxis={'side': 'bottom'})

        fig_objs.append(fig)
    return fig_objs


def plot_classification_report(yTrue: pd.Series, yPred: pd.Series, model_names: List):
    """
    Create classification report in table form

    Arguments:
        yTrue (:obj:`pd.Series`):
            true labels, output from int_general_metrics
        yPred (:obj:`pd.Series`):
            predicted labels, output from int_general_metrics
        model_names (:obj:`List[str]`):
            model names, output from interpreter int_general_metrics

    Returns:
        :obj:`List[~plotly.graph_objects.Figure]`:
            list of tables displaying classification report details
    """  
    fig_objs = []
    for i in range(len(model_names)):
        cls_rpt = classification_report(yTrue, yPred[i], output_dict=True)
        cls_rpt_df = pd.DataFrame(cls_rpt).transpose()

        for j, ind in enumerate(list(cls_rpt_df.index)):
            if ind == 'accuracy':
                """
                To better display the accuracy record
                """
                cls_rpt_df.iloc[j, 0:2] = ''
                cls_rpt_df.iloc[j, -1] = cls_rpt_df.iloc[j + 1, -1]

        header = [''] + cls_rpt_df.columns.tolist()
        values_cells = [cls_rpt_df.index.tolist(), 
                        [f'{i:.4f}' if i != '' else i for i in cls_rpt_df['precision']],
                        [f'{i:.4f}' if i != '' else i for i in cls_rpt_df['recall']],
                        [f'{i:.4f}' for i in cls_rpt_df['f1-score']],
                        cls_rpt_df['support'].tolist()]

        fig = go.Figure(data=[go.Table(header=dict(values=header), cells=dict(values=values_cells, height=28))])
        # height => cell height

        fig.update_layout(
            title=f'Classification Report : <b>{model_names[i]}</b>', 
            title_x=0.5, 
            autosize=True,
            margin={'b': 0, 'pad': 4})
        fig_objs.append(fig)
    return fig_objs


def plot_roc_curve(yTrue: pd.Series, yPred: pd.Series, model_names: List):
    """
    Display roc curve for comparison on various models

    Arguments:
        yTrue (:obj:`pd.Series`):
            true labels, output from int_general_metrics
        yPred (:obj:`pd.Series`):
            predicted labels, output from int_general_metrics
        model_names (:obj:`List[str]`):
            model names, output from interpreter int_general_metrics

    Returns:
        :obj:`~plotly.graph_objects.Figure`:
            figure displaying line curves comparing roc-auc score for various models
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='navy', dash='dash'), showlegend=False))

    is_multiclass = len(set(yTrue)) > 2
    if is_multiclass:
        for i in range(len(model_names)):
            fpr_j_list = []
            tpr_j_list = []
            score_j_list = []
            for j in sorted(set(yTrue)):
                yTrue_j = [1 if x == j else 0 for x in yTrue]
                yPred_j = [pred[1] if pred[0] == j else 1 - pred[1] for pred in yPred[i]]

                fpr_j, tpr_j, _ = roc_curve(yTrue_j, yPred_j)
                score_j = roc_auc_score(yTrue_j, yPred_j)
                fpr_j_list.append(fpr_j)
                tpr_j_list.append(tpr_j)
                score_j_list.append(score_j)

            for k in range(len(fpr_j_list)):
                """
                To plot roc_curve on same figure for multiple models
                """
                fig.add_trace(go.Scatter(
                    x=fpr_j_list[k], 
                    y=tpr_j_list[k], 
                    mode='lines',
                    name=f'model_{model_names[i]}_class_{list(sorted(set(yTrue)))[k]} [score: {score_j_list[k]:.4f}]', 
                    hoverlabel=dict(namelength=-1)))
    else:
        fpr_list = []
        tpr_list = []
        score_list = []
        yTrue = [int(i) for i in yTrue]
        for i in range(len(yPred)):
            fpr, tpr, _ = roc_curve(yTrue, yPred[i])
            score = roc_auc_score(yTrue, yPred[i])
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            score_list.append(score)

        for i in range(len(yPred)):
            """
            To plot roc_curve on same figure for multiple models
            """
            fig.add_trace(go.Scatter(
                x=fpr_list[i], 
                y=tpr_list[i], 
                mode='lines', 
                name=f'{model_names[i]} [score: {score_list[i]:.4f}]', 
                hoverlabel=dict(namelength=-1)))

    fig.update_layout(title='<b>Receiver Operating Characteristic (ROC)</b>', 
                      xaxis_title='False Positive Rate', 
                      yaxis_title='True Positive Rate', 
                      title_x=0.2, 
                      width=500)
    return fig


def plot_precisionRecall_curve(yTrue: pd.Series, yPred: pd.Series, model_names: List):
    """
    Display precision-recall curve for comparison on various models

    Arguments:
        yTrue (:obj:`pd.Series`):
            true labels, output from int_general_metrics
        yPred (:obj:`pd.Series`):
            predicted labels, output from int_general_metrics
        model_names (:obj:`List[str]`):
            model names, output from interpreter int_general_metrics

    Returns:
        :obj:`~plotly.graph_objects.Figure`:
            figure displaying line curves comparing precision-recall for various models
    """
    fig = go.Figure()

    is_multiclass = len(set(yTrue)) > 2
    if is_multiclass:
        for i in range(len(model_names)):
            precision_j_list = []
            recall_j_list = []
            score_j_list = []
            for j in sorted(set(yTrue)):
                yTrue_j = [1 if x == j else 0 for x in yTrue]
                yPred_j = [pred[1] if pred[0] == j else 1 - pred[1] for pred in yPred[i]]

                precision, recall, _ = precision_recall_curve(yTrue_j, yPred_j)
                score = average_precision_score(yTrue_j, yPred_j)
                precision_j_list.append(precision)
                recall_j_list.append(recall)
                score_j_list.append(score)

            for k in range(len(precision_j_list)):
                """
                To plot curves on same figure for multiple models
                """
                fig.add_trace(go.Scatter(
                    x=recall_j_list[k], 
                    y=precision_j_list[k], 
                    fill='tozeroy', 
                    name=f'model_{model_names[i]}_class{list(sorted(set(yTrue)))[k]} [score: {score_j_list[k]:.4f}]', 
                    hoverlabel=dict(namelength=-1)))
    else:
        precision_list = []
        recall_list = []
        score_list = []
        yTrue = [int(i) for i in yTrue]
        for i in range(len(yPred)):
            precision, recall, _ = precision_recall_curve(yTrue, yPred[i])
            score = average_precision_score(yTrue, yPred[i])
            precision_list.append(precision)
            recall_list.append(recall)
            score_list.append(score)

        for i in range(len(yPred)):
            """
            To plot curves on same figure for multiple models
            """
            fig.add_trace(go.Scatter(
                x=recall_list[i], 
                y=precision_list[i], 
                fill='tozeroy', 
                name=f'model_{model_names[i]} [score: {score_list[i]:.4f}]', 
                hoverlabel=dict(namelength=-1)))

    fig.update_layout(title='<b>Precision Recall Curve</b>', 
                      xaxis_title='Recall', 
                      yaxis_title='Precision', 
                      title_x=0.4,
                      showlegend=True)
    return fig


def plot_prediction_vs_actual(df: pd.DataFrame):
    '''
    Display scatter plot for comparison on actual values vs prediction values

    Arguments:
        df (:obj:`pd.DataFrame`):
            dataframe containing yTrue and yPred values, output from int_general_metrics

    Returns:
        :obj:`~plotly.graph_objects.Figure`:
            figure displaying scatter plot comparing actual values vs prediction values
    '''
    def _modify_legend_name(fig, legend_name_dict):
        for i, dt in enumerate(fig.data):
            for element in dt:
                if element == 'name':
                    fig.data[i].name = legend_name_dict[fig.data[i].name]
        return fig

    pred_cols = [col for col in df.columns if 'yPred_' in col]
    corrected_legend_names = [col.replace('yPred_', '') for col in pred_cols]
    legend_name_dict = dict(zip(pred_cols, corrected_legend_names))

    fig = px.scatter(df, x='yTrue', y=pred_cols,
                    trendline='ols',
                    marginal_x='histogram',
                    marginal_y='histogram',
                    color_discrete_sequence=px.colors.qualitative.D3)
    fig.update_layout(
        title='<b>Comparison of Prediction (yPred) vs Actual (yTrue)</b>',
        title_x=0.12,
        xaxis_title="Actual",
        yaxis_title="Prediction",
        legend_title="", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
        margin=dict(t=110, l=30, r=30))

    # scatter plot for pred_cols[0]
    hv_template0_str = f'<b>{legend_name_dict[pred_cols[0]]}</b>'
    fig.data[0].hovertemplate = hv_template0_str + '<br><br>Actual : %{x}<br>Prediction : %{y}<extra></extra>'
    # histogram-x
    fig.data[1].hovertemplate = fig.data[1].hovertemplate.replace('value', 'yTrue').replace('variable=yPred_lasso', '')
    # histogram-y
    fig.data[2].hovertemplate = fig.data[2].hovertemplate.replace('value', 'yPred').replace('variable=yPred_lasso', '')
    # trendline for scatter plot data[0]
    fig.data[3].hovertemplate = fig.data[3].hovertemplate.replace('value', 'yPred').replace('variable=yPred_', '')

    if len(pred_cols) > 1:  # Bimodal
        # scatter plot for pred_cols[1]
        hv_template2_str = f'{legend_name_dict[pred_cols[1]]}'
        fig.data[4].hovertemplate = hv_template2_str + '<br><br>Actual : %{x}<br>Prediction : %{y}<extra></extra>'
        # histogram-x
        text_to_replace = 'variable=yPred_random_forest'
        fig.data[5].hovertemplate = fig.data[5].hovertemplate.replace('value', 'yTrue').replace(text_to_replace, '')
        # histogram-y
        fig.data[6].hovertemplate = fig.data[6].hovertemplate.replace('value', 'yPred').replace(text_to_replace, '')
        # trendline for scatter plot data[1]
        fig.data[7].hovertemplate = fig.data[7].hovertemplate.replace('value', 'yPred').replace('variable=yPred_', '')

    fig = _modify_legend_name(fig=fig, legend_name_dict=dict(zip(pred_cols, corrected_legend_names)))
    return fig


def plot_prediction_offset_overview(df: pd.DataFrame):
    '''
    Display scatter plot for overview on prediction offset values

    Arguments:
        df (:obj:`~pd.DataFrame`):
            dataframe containing yTrue and yPred values, output from int_general_metrics

    Returns:
        :obj:`~plotly.graph_objects.Figure`:
            figure displaying scatter plot outlining overview on prediction offset values
    '''
    pred_cols = [col for col in df.columns if 'yPred_' in col]
    corrected_legend_names = [col.replace('yPred_', '') for col in pred_cols]
    legend_name_dict = dict(zip(pred_cols, corrected_legend_names))
    max_range = int(df[pred_cols].max().max())

    offset_cols = []
    for col in pred_cols:
        offset_col = f'offset_{legend_name_dict[col]}'
        df[offset_col] = df[col] - df['yTrue']
        offset_cols.append(offset_col)

    fig = px.scatter(df, x=pred_cols[0], y=offset_cols[0], color_discrete_sequence=px.colors.qualitative.D3)
    fig.data[0].name = corrected_legend_names[0]
    fig.update_traces(showlegend=True, hovertemplate="Prediction : %{x}<br>Offset : %{y}")

    if len(pred_cols) > 1:  # Bimodal
        fig.add_trace(go.Scatter(
            x=df[pred_cols[1]], 
            y=df[offset_cols[1]], 
            name=corrected_legend_names[1], 
            mode='markers',
            marker=dict(color='#FF7F0E'),
            hovertemplate="Prediction : %{x}<br>Offset : %{y}"))

    # add reference baseline [mainly to have baseline included in legend]
    fig.add_trace(go.Scatter(
        x=[0, max_range], 
        y=[0] * 2, 
        name="Baseline [Prediction - Actual]", 
        visible=True, 
        hoverinfo='skip',
        mode='lines',
        line=dict(color="green", dash="dot")))
    # referece baseline [mainly for the dotted line in graph, but no legend generated]
    fig.add_hline(y=0, line_dash="dot")

    fig.update_layout(
                title='<b>Prediction Offset Overview</b>', 
                xaxis_title='Prediction', 
                yaxis_title='Offset from baseline', 
                title_x=0.3,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
                margin=dict(t=110, l=30, r=30))

    return fig


def plot_std_error_metrics(df: pd.DataFrame):
    '''
    Display table comparing various standard metrics for regression task

    Arguments:
        df (:obj:`~pd.DataFrame`):
            dataframe containing info on error metrics, output from int_general_metrics

    Returns:
        :obj:`~dash_table.DataTable`:
            table object comparing various standard metrics for regression task
    '''
    fig = dash_table.DataTable(
        id='table', 
        columns=[{'id': c, 'name': c} if c != 'Formula' else {'id': c, 'name': c, 'presentation': 'markdown'} for c in df.columns], 
        style_cell={'textAlign': 'center', 'border': '1px solid rgb(229, 211, 197)', 'font-family': 'Arial', 'margin-bottom': '0'},
        style_header={'fontWeight': 'bold', 'color': 'white', 'backgroundColor': '#7e746d ', 'border': '1px solid rgb(229, 211, 197)'},
        style_table={'width': '98%', 'margin': 'auto'},
        data=df.to_dict('records'))
    return fig

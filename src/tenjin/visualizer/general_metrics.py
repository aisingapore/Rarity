# import warnings
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go


def plot_confusion_matrix(yTrue, yPred, model_names):
    """
    Create confusion matrix
    
    Arguments:
        yTrue, yPred, model_names -- output from interpreter [int_general_metrics]

    Returns:
        plotly graph -- displaying confusion matrix details
    """
    fig_objs = []
    for i in range(len(yPred)):
        conf_matrix = confusion_matrix(yTrue, yPred[i], labels=list(sorted(set(yTrue))))
    
        fig = px.imshow(conf_matrix, 
                        labels=dict(x="Predicted Label", y="True Label", color="No.of Label"),
                        color_continuous_scale=px.colors.sequential.Viridis,
                        x=list(sorted(set(yPred[i]))), # x = y
                        y=list(sorted(set(yTrue))), # y = x
                        title=f'Confusion Matrix : <b>{model_names[i]}</b>') 
        fig.update_layout(title={'y':0.90, 
                                 'x':0.5, 
                                 'xanchor': 'center', 
                                 'yanchor': 'top', 
                                 },
                          margin=dict(r=180),
                          xaxis={'side':'bottom'})

        fig_objs.append(fig)
    return fig_objs


def plot_classification_report(yTrue, yPred, model_names):
    """
    Create classification report in table form
    
    Arguments:
        yTrue, yPred, model_names -- output from interpreter [int_general_metrics]
    
    Returns:
        plotly table -- containing classification report details
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


def plot_roc_curve(yTrue, yPred, model_names):
    """
    Display roc curve for comparison on various models
    
    Arguments:
        yTrue, yPred, model_names -- output from interpreter [int_general_metrics]
    
    Returns:
        plotly line curve -- comparing roc-auc score for various models
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
                yTrue_j = [1 if x==j else 0 for x in yTrue]
                yPred_j = [pred[1] if pred[0]==j else 1-pred[1] for pred in yPred[i]]
                
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
                    hoverlabel = dict(namelength = -1)))
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
                hoverlabel = dict(namelength = -1)))

    fig.update_layout(title='<b>Receiver Operating Characteristic (ROC)</b>', 
                      xaxis_title='False Positive Rate', 
                      yaxis_title='True Positive Rate', 
                      title_x=0.2, 
                      width=500)
    return fig


def plot_precisionRecall_curve(yTrue, yPred, model_names):
    """
    Display precision-recall curve for comparison on various models
    
    Arguments:
        yTrue, yPred, model_names -- output from interpreter [int_general_metrics]
    
    Returns:
        plotly line curve -- comparing roc-auc score for various models
    """
    fig = go.Figure()
    
    is_multiclass = len(set(yTrue)) > 2
    if is_multiclass:
        for i in range(len(model_names)):
            precision_j_list = []
            recall_j_list = []
            score_j_list = []
            for j in sorted(set(yTrue)):
                yTrue_j = [1 if x==j else 0 for x in yTrue]
                yPred_j = [pred[1] if pred[0]==j else 1-pred[1] for pred in yPred[i]]
                
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
                    hoverlabel = dict(namelength = -1)))
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
                hoverlabel=dict(namelength = -1)))

    fig.update_layout(title='<b>Precision Recall Curve</b>', 
                      xaxis_title='Recall', 
                      yaxis_title='Precision', 
                      title_x=0.4,
                      showlegend=True)
    return fig
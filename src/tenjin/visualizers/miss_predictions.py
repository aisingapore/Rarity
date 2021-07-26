# import plotly.graph_objects as go
import plotly.express as px
import dash_table


def plot_probabilities_spread_pattern(df_specific_label):
    '''
    Display scatter plot for probabilities comparison on correct data point vs miss-predicted data point
    for each class label

    Arguments:
        df -- output from interpreter [int_miss_predictions], dataframe of 1 specific label of 1 model type

    Returns:
        plotly scatter plot 
    '''
    label = list(df_specific_label.columns)[1]
    model_name = df_specific_label['model'].values[0]

    fig = px.scatter(df_specific_label, 
                     x=list(df_specific_label.index), 
                     y=df_specific_label[label], 
                     color='pred_state', 
                     category_orders={"pred_state": ["correct", "miss-predict"]})

    fig.update_layout(
        title=f'<b>Class {label}<br>[ {model_name} ]</b><br>',
        title_x=0.6,
        yaxis_title=f"Probability is_class_{label} ",
        yaxis_showgrid=False,
        xaxis_title="data_point index",
        xaxis_showgrid=False,
        legend_title="", 
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=0.8), 
        width=250,
        height=600,
        margin=dict(t=170, b=0, l=12, r=12, pad=10))

    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(rangemode="tozero")
    fig.add_hline(y=0.5, line_dash="dot")

    # iterate through all traces, to ensure all label-class have consistent format
    for i in range(len(fig.data)):
        if fig.data[i]['legendgroup'] == 'correct':
            fig.data[i]['marker']['color'] = '#636efa' 
            fig.data[i]['hovertemplate'] = "<b>Index %{x}</b><br>" + "<b>[ correct ]</b><br><br>" + \
                                            "probability: %{y:.4f}<br>" + "<extra></extra>"

        elif fig.data[i]['legendgroup'] == 'miss-predict':
            fig.data[i]['marker']['color'] = '#EF553B'
            fig.data[i]['hovertemplate'] = "<b>Index %{x}</b><br>" + "<b>[ miss-predict ]</b><br><br>" + \
                                            "probability: %{y:.4f}<br>" + "<extra></extra>"
    return fig


def plot_simple_probs_spread_overview(df_label_state):
    '''
    Display data table listing simple stats on ss, %correct, % wrong, accuracy for each label class

    Arguments:
        df -- output from interpreter [int_general_metrics]

    Returns:
        dash table
    '''
    fig = dash_table.DataTable(
        id='table', 
        columns=[{'id': c, 'name': c} for c in df_label_state.columns], 
        style_cell={'font-family': 'verdana', 
                    'font-size': '14px', 
                    'border': 'none', 
                    'minWidth': '100px'},
        style_header={'display': 'none'},
        style_table={'width': '550', 'margin': 'auto'},
        style_data={'lineHeight': '15px'},
        style_data_conditional=[{'if': {'column_id': 'index'}, 'textAlign': 'left'},
                                {'if': {'column_id': 'state_value'}, 'textAlign': 'right'}],
        data=df_label_state.to_dict('records'))
    return fig
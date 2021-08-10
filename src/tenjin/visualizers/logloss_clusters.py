import plotly.graph_objects as go


def plot_logloss_clusters(df, cluster_score_ls):
    models = [col.replace('cluster_', '') for col in df.columns if 'cluster_' in col]

    fig = go.Figure()
    fig.add_trace(go.Violin(x=df[f'cluster_{models[0]}'],
                            y=df[f'offset_{models[0]}'],
                            legendgroup=models[0], scalegroup=models[0], name=models[0],
                            line_color='#1f77b4',
                            customdata=list(df.index),
                            hovertemplate='index=%{customdata}<br>cluster=%{x}<br>offset=%{y}<br>',
                            showlegend=True))

    if len(models) > 1:
        fig.add_trace(go.Violin(x=df[f'cluster_{models[1]}'],
                                y=df[f'offset_{models[1]}'],
                                legendgroup=models[1], scalegroup=models[1], name=models[1],
                                line_color='#FF7F0E',
                                customdata=list(df.index),
                                hovertemplate='index=%{customdata}<br>cluster=%{x}<br>offset=%{y}<br>',
                                showlegend=True))

    # update characteristics shared by all traces
    fig.update_traces(meanline_visible=True,
                    box_visible=True,
                    points='all',  # show all points
                    jitter=0.05,  # add some jitter on points for better visibility
                    scalemode='count')  # scale violin plot area with total count

    fig.update_layout(
        title_text='<b>Overview of Prediction Offset Clusters</b>',
        title_x=0.5,
        xaxis_title='Cluster',
        yaxis_title='Offset from baseline',
        width=1250,
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        violingap=0.2, violingroupgap=0.3, violinmode='overlay')

    return fig

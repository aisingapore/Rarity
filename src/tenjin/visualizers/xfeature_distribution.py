import plotly.express as px


def plot_distribution_by_specific_feature(specific_feature_ls, kl_div_dict_sorted):
    if isinstance(specific_feature_ls, list):
        pass
    else:
        specific_feature_ls = [specific_feature_ls]

    fig_obj_ls = []
    for specific_feature in specific_feature_ls:
        df = kl_div_dict_sorted[specific_feature][1]
        fig = px.histogram(df, x=specific_feature, color='dataset_type',
                            marginal='box', opacity=0.5, barmode='overlay',
                            color_discrete_sequence=px.colors.qualitative.D3)
        kl_div_score = kl_div_dict_sorted[specific_feature][0]

        fig.update_layout(
                title=f'<b>[ KL divergence : {kl_div_score:.4f} ]</b><br>Distribution of xFeature : {specific_feature}',
                xaxis_title=f'xFeature : {specific_feature}', 
                yaxis_title='count', 
                title_x=0.45,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
                margin=dict(t=120))
        fig_obj_ls.append(fig)
    return fig_obj_ls


def plot_distribution_by_kl_div_ranking(kl_div_dict_sorted, display_option, display_value):
    fig_obj_dict = {}

    if display_option == 'top':
        sliced_kl_div_dict_top = dict(list(kl_div_dict_sorted.items())[:display_value])
        fig_obj_dict['top'] = [plot_distribution_by_specific_feature(k, sliced_kl_div_dict_top) for k, v in sliced_kl_div_dict_top.items()]
        fig_obj_dict['bottom'] = []
    elif display_option == 'bottom':
        sliced_kl_div_dict_btm = dict(list(kl_div_dict_sorted.items())[-display_value:])
        fig_obj_dict['top'] = []
        fig_obj_dict['bottom'] = [plot_distribution_by_specific_feature(k, sliced_kl_div_dict_btm) for k, v in sliced_kl_div_dict_btm.items()]
    elif display_option == 'both':
        sliced_kl_div_dict_top = dict(list(kl_div_dict_sorted.items())[:display_value])
        sliced_kl_div_dict_btm = dict(list(kl_div_dict_sorted.items())[-display_value:])
        fig_obj_dict['top'] = [plot_distribution_by_specific_feature(k, sliced_kl_div_dict_top) for k, v in sliced_kl_div_dict_top.items()]
        fig_obj_dict['bottom'] = [plot_distribution_by_specific_feature(k, sliced_kl_div_dict_btm) for k, v in sliced_kl_div_dict_btm.items()]
    return fig_obj_dict

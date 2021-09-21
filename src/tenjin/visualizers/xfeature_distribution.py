import plotly.express as px


def plot_distribution_by_specific_feature(ls_specific_feature, kl_div_dict_sorted, comparison_base, model_name):
    if not isinstance(ls_specific_feature, list):
        ls_specific_feature = [ls_specific_feature]

    fig_obj_ls = []
    for specific_feature in ls_specific_feature:
        fig = _single_dist_plot(specific_feature, kl_div_dict_sorted, comparison_base, model_name)
        fig_obj_ls.append(fig)
    return fig_obj_ls


def plot_distribution_by_kl_div_ranking(kl_div_dict_sorted, display_option, display_value, comparison_base, model_name):
    fig_obj_dict = {}

    if display_option == 'top':
        sliced_kl_div_dict_top = dict(list(kl_div_dict_sorted.items())[:display_value])
        fig_obj_dict['top'] = [plot_distribution_by_specific_feature(k, sliced_kl_div_dict_top, comparison_base, model_name)
                                for k, v in sliced_kl_div_dict_top.items()]
        fig_obj_dict['bottom'] = []

    elif display_option == 'bottom':
        sliced_kl_div_dict_btm = dict(list(kl_div_dict_sorted.items())[-display_value:])
        fig_obj_dict['top'] = []
        fig_obj_dict['bottom'] = [plot_distribution_by_specific_feature(k, sliced_kl_div_dict_btm, comparison_base, model_name)
                                    for k, v in sliced_kl_div_dict_btm.items()]

    elif display_option == 'both':
        sliced_kl_div_dict_top = dict(list(kl_div_dict_sorted.items())[:display_value])
        sliced_kl_div_dict_btm = dict(list(kl_div_dict_sorted.items())[-display_value:])
        fig_obj_dict['top'] = [plot_distribution_by_specific_feature(k, sliced_kl_div_dict_top, comparison_base, model_name)
                                for k, v in sliced_kl_div_dict_top.items()]
        fig_obj_dict['bottom'] = [plot_distribution_by_specific_feature(k, sliced_kl_div_dict_btm, comparison_base, model_name)
                                    for k, v in sliced_kl_div_dict_btm.items()]
    return fig_obj_dict


def _single_dist_plot(feature, kl_div_dict_sorted, comparison_base, model_name):
    '''
    comparison_base :
    for regression  => dataset_type
    for classification => pred_state
    '''
    df = kl_div_dict_sorted[feature][1]

    fig = px.histogram(df, x=feature, color=comparison_base,
                        marginal='box', opacity=0.5, barmode='overlay',
                        color_discrete_sequence=px.colors.qualitative.D3)

    kl_div_score = kl_div_dict_sorted[feature][0]

    customized_title = f'<b>[ KL divergence : {kl_div_score:.4f} ]</b><br>Distribution of xFeature : {feature}'
    customized_margin = dict(t=120)
    if model_name is not None:
        model_name = f'<span style="color:blue; font-size:14px">{model_name}   </span>'
        feature_name = f'<span style="font-size:14px">Distribution of xFeature : {feature}</span>'
        customized_title = f'<b>[ KL divergence : {kl_div_score:.4f} ]</b><br>' + model_name + feature_name
        customized_margin = dict(t=130)

    fig.update_layout(
            title=customized_title,
            xaxis_title=f'xFeature : {feature}',
            yaxis_title='count',
            title_x=0.48,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
            margin=customized_margin)
    return fig

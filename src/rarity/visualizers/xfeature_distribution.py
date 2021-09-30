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

from typing import List, Dict
import plotly.express as px


def plot_distribution_by_specific_feature(ls_specific_feature: List[str], kl_div_dict_sorted: Dict, comparison_base: str, model_name: str):
    '''
    Create distribution plot for a specific feature

    Arguments:
        ls_specific_feature (:obj:`List[str]`):
            list of feature to have its distribution graph plotted
        kl_div_dict_sorted (:obj:`Dict`):
            dictionary storing kl-divergence score by feature in decending order
        comparison_base (str):
            info to indicate the baseline for distribution comparison. ``dataset_type`` for regression and ``pred_state`` for classification task
        model_name (str):
            model used to generate yPred

    Returns:
        :obj:`List[~plotly.graph_objects.Figure]`:
            List of figures displaying distribution plot of specific feature
    '''
    if not isinstance(ls_specific_feature, list):
        ls_specific_feature = [ls_specific_feature]

    fig_obj_ls = []
    for specific_feature in ls_specific_feature:
        fig = _single_dist_plot(specific_feature, kl_div_dict_sorted, comparison_base, model_name)
        fig_obj_ls.append(fig)
    return fig_obj_ls


def plot_distribution_by_kl_div_ranking(kl_div_dict_sorted: Dict, display_option: str, display_value: int,
                                        comparison_base: str, model_name: str):
    '''
    Create distribution plot by kl-divergence score ranking in descending order

    Arguments:
        kl_div_dict_sorted (:obj:`Dict`):
            dictionary storing kl-divergence score by feature in decending order
        display_option (str)
            - info to indicate if to display distribution plot by top-N / bottom-N or both top-N + bottom-N
            - Available options: ``top``, ``bottom`` or ``both``
        display_value (int)
            - number indicates the limit of graph to be displayed, max at 10
            - if dataset consists of < 10 features, the limit == no. of features the dataset has
        comparison_base (str):
            info to indicate the baseline for distribution comparison. ``dataset_type`` for regression and ``pred_state`` for classification task
        model_name (str):
            model used to generate yPred

    Returns:
        :obj:`Dict[str, ~plotly.graph_objects.Figure]`:
            Dictionary storing distribution figures by display_option
    '''
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


def _single_dist_plot(feature: str, kl_div_dict_sorted: Dict, comparison_base: str, model_name: str):
    '''
    Internal function to plot single distribution graph

    Important Arguments:

        comparison_base (str):
        ``dataset_type`` for regression, ``pred_state`` for classification
    '''
    df = kl_div_dict_sorted[feature][1]

    fig = px.histogram(df, x=feature, color=comparison_base,
                        marginal='box', opacity=0.5, barmode='overlay',
                        # color_discrete_sequence=px.colors.qualitative.D3)
                        # replaces default color mapping by value
                        color_discrete_map={'correct': '#1F77B4', 'miss-predict': '#FF7F0E', 'df_reference': '#1F77B4', 'df_sliced': '#FF7F0E'},
                        category_orders={'pred_state': ['correct', 'miss-predict'], 'dataset_type': ['df_reference', 'df_sliced']})

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

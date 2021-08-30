import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from tenjin.app import app
from tenjin.interpreters.structured_data import IntFeatureDistribution
from tenjin.visualizers.xfeature_distribution import plot_distribution_by_specific_feature, plot_distribution_by_kl_div_ranking
from tenjin.utils import style_configs
from tenjin.utils.common_functions import detected_bimodal, invalid_slicing_range, invalid_slicing_range_sequence, \
                                            invalid_min_max_limit, incomplete_range_entry


def fig_plot_distribution_by_kl_div_ranking(data_loader, feature_to_exclude, start_idx, stop_idx, display_option, display_value):
    if data_loader.analysis_type == 'regression':
        kl_div_dict_sorted = IntFeatureDistribution(data_loader).xform(feature_to_exclude, start_idx, stop_idx)
        fig_obj_dict = plot_distribution_by_kl_div_ranking(kl_div_dict_sorted, display_option, display_value, 'dataset_type', None)
        return fig_obj_dict

    elif 'classification' in data_loader.analysis_type:
        models = data_loader.get_model_list()
        ls_kl_div_dict_sorted = IntFeatureDistribution(data_loader).xform(feature_to_exclude, start_idx, stop_idx)
        ls_fig_obj_dict = [plot_distribution_by_kl_div_ranking(kl_div_dict, display_option, display_value, 'pred_state', models[i])
                            for i, kl_div_dict in enumerate(ls_kl_div_dict_sorted)]
        return ls_fig_obj_dict


def fig_plot_distribution_by_specific_feature(data_loader, ls_specific_feature, start_idx, stop_idx):
    if data_loader.analysis_type == 'regression':
        kl_div_dict_sorted = IntFeatureDistribution(data_loader).xform(None, start_idx, stop_idx)
        fig_objs_specific = plot_distribution_by_specific_feature(ls_specific_feature, kl_div_dict_sorted, 'dataset_type', None)
        return fig_objs_specific

    elif 'classification' in data_loader.analysis_type:
        models = data_loader.get_model_list()
        ls_kl_div_dict_sorted = IntFeatureDistribution(data_loader).xform(None, start_idx, stop_idx)
        ls_fig_objs_specific = [plot_distribution_by_specific_feature(ls_specific_feature, kl_div_dict, 'pred_state', models[i])
                                for i, kl_div_dict in enumerate(ls_kl_div_dict_sorted)]
        return ls_fig_objs_specific


def _fig_layout_top_n(fig_obj_dict, display_option, model_pair=True):
    fig_objs_display_combined = []
    if model_pair:
        for n in range(len(fig_obj_dict[0][display_option])):
            shared_title_top_n = html.H4(f'Top {n + 1}', className='h4__title-kl-div-rank-num-top-modelpair')
            fig_objs_display_combined.append(shared_title_top_n)

            top_n_fig_cls_m1 = fig_obj_dict[0][display_option][n][0]
            top_n_fig_cls_m2 = fig_obj_dict[1][display_option][n][0]
            single_distplot_top_pair = [dbc.Row([
                                            dbc.Col([
                                                dbc.Row(dbc.Col(dcc.Graph(id=f'fig-top{n + 1}-feat-dist-cls-m1', figure=top_n_fig_cls_m1)),
                                                justify='center', className='border__kl-div-figs-both-left')], width=6),
                                            dbc.Col([
                                                dbc.Row(dbc.Col(dcc.Graph(id=f'fig-top{n + 1}-feat-dist-cls-m2', figure=top_n_fig_cls_m2)),
                                                justify='center', className='border__kl-div-figs-both-right')])
                                        ])]
            fig_objs_display_combined += single_distplot_top_pair
    else:
        for n in range(len(fig_obj_dict[display_option])):
            top_n_fig_reg = fig_obj_dict[display_option][n][0]
            single_distplot_top = [html.H4(f'Top {n + 1}', className='h4__title-kl-div-rank-num-top'),
                                    dbc.Row([
                                        dbc.Row(dcc.Graph(id=f'fig-top{n + 1}-feat-dist-reg', figure=top_n_fig_reg), justify='center')],
                                        justify='center', className='border__kl-div-figs')]
            fig_objs_display_combined += single_distplot_top
    return fig_objs_display_combined


def _fig_layout_bottom_n(fig_obj_dict, display_option, model_pair=True):
    fig_objs_display_combined = []
    if model_pair:
        for n in range(len(fig_obj_dict[0][display_option])):
            shared_title_bottom_n = html.H4(f'Bottom {n + 1}', className='h4__title-kl-div-rank-num-bottom-modelpair')
            fig_objs_display_combined.append(shared_title_bottom_n)

            btm_n_fig_cls_m1 = fig_obj_dict[0][display_option][-(n + 1)][0]  # n + 1 => to avoid -0
            btm_n_fig_cls_m2 = fig_obj_dict[1][display_option][-(n + 1)][0]  # n + 1 => to avoid -0
            single_distplot_bottom_pair = [dbc.Row([
                                            dbc.Col([
                                                dbc.Row(dbc.Col(dcc.Graph(id=f'fig-bottom{n + 1}-feat-dist-cls-m1', figure=btm_n_fig_cls_m1)),
                                                justify='center', className='border__kl-div-figs-both-left')], width=6),
                                            dbc.Col([
                                                dbc.Row(dbc.Col(dcc.Graph(id=f'fig-bottom{n + 1}-feat-dist-cls-m2', figure=btm_n_fig_cls_m2)),
                                                justify='center', className='border__kl-div-figs-both-right')])
                                        ])]
            fig_objs_display_combined += single_distplot_bottom_pair
    else:
        for n in range(len(fig_obj_dict[display_option])):
            btm_n_fig_reg = fig_obj_dict[display_option][-(n + 1)][0]  # n + 1 => to avoid -0
            single_distplot_btm = [html.H4(f'Bottom {n + 1}', className='h4__title-kl-div-rank-num-bottom'),
                                    dbc.Row([
                                        dbc.Row(dcc.Graph(id=f'fig-bottom{n + 1}-feat-dist-reg', figure=btm_n_fig_reg), justify='center')],
                                        justify='center', className='border__kl-div-figs')]
            fig_objs_display_combined += single_distplot_btm
    return fig_objs_display_combined


def _model_pair_fig_objs_layout_on_display_option(ls_fig_obj_dict, display_option):
    fig_objs_pairs_display_combined = []
    if display_option == 'top':
        fig_objs_pairs_display_combined = _fig_layout_top_n(ls_fig_obj_dict, display_option, model_pair=True)

    elif display_option == 'bottom':
        fig_objs_pairs_display_combined = _fig_layout_bottom_n(ls_fig_obj_dict, display_option, model_pair=True)

    elif display_option == 'both':
        fig_objs_pairs_display_combined = _fig_layout_top_n(ls_fig_obj_dict, 'top', model_pair=True) + \
                                            _fig_layout_bottom_n(ls_fig_obj_dict, 'bottom', model_pair=True)
    return fig_objs_pairs_display_combined


def _fig_objs_layout_based_on_display_option(fig_obj_dict, display_option):
    '''
    fig_obj_dict => already filtered to top n value selected by user
    '''
    fig_objs_display_combined = []
    if display_option == 'top':
        fig_objs_display_combined = _fig_layout_top_n(fig_obj_dict, display_option, model_pair=False)

    elif display_option == 'bottom':
        fig_objs_display_combined = _fig_layout_bottom_n(fig_obj_dict, display_option, model_pair=False)

    elif display_option == 'both':
        fig_objs_display_combined = _fig_layout_top_n(fig_obj_dict, 'top', model_pair=False) + \
                                    _fig_layout_bottom_n(fig_obj_dict, 'bottom', model_pair=False)
    return fig_objs_display_combined


def _fig_objs_layout_based_on_specific_feature(specific_feature, fig_objs_specific):
    fig_objs_specific_feature_combined = []
    if len(specific_feature) > 0:
        for i, specific_fig in enumerate(fig_objs_specific):
            single_displot_specific = [dbc.Row(dcc.Graph(id=f'fig-specific{i}-feat-dist-reg', figure=specific_fig), 
                                                justify='center', className='border__specific-feat-dist')]
            fig_objs_specific_feature_combined += single_displot_specific
    return fig_objs_specific_feature_combined


def _model_pair_fig_objs_layout_on_specific_feature(specific_feature, ls_fig_objs_specific):
    fig_objs_specific_feature_combined = []
    if len(specific_feature) > 0:
        for i, (specific_fig_m1, specific_fig_m2) in enumerate(zip(ls_fig_objs_specific[0], ls_fig_objs_specific[1])):
            single_displot_specific_model_pair = [dbc.Row([
                                                        dbc.Col(dbc.Row([
                                                                    dbc.Col(dcc.Graph(id=f'fig-specific{i}-feat-dist-cls-m1',
                                                                                        figure=specific_fig_m1))],
                                                                justify='center', className='border__kl-div-figs-both-left'), width=6),
                                                        dbc.Col(dbc.Row([
                                                                    dbc.Col(dcc.Graph(id=f'fig-specific{i}-feat-dist-cls-m2',
                                                                                        figure=specific_fig_m2))],
                                                                justify='center', className='border__kl-div-figs-both-right'), width=6)])]

            fig_objs_specific_feature_combined += single_displot_specific_model_pair
    return fig_objs_specific_feature_combined


def _second_level_validity_check_on_user_input_range(range_list, df):
    range_input_err_alert = ''
    if incomplete_range_entry(range_list):
        range_input_err_alert = style_configs.activate_incomplete_range_entry_alert()
    elif invalid_slicing_range_sequence(range_list):
        range_input_err_alert = style_configs.activate_range_input_error_alert()
    elif invalid_min_max_limit(range_list, df):
        range_input_err_alert = style_configs.activate_invalid_limit_alert(df)
    return range_input_err_alert


def _get_max_value_on_slider(df_features):
    max_value = len(list(df_features.columns))
    if len(list(df_features.columns)) > 10:
        max_value = 10
    return max_value


def _generate_valid_start_stop_idx_based_on_usr_input_range(slicing_input, df_features, default_start_idx, default_stop_idx):
    if slicing_input is not None:
        if invalid_slicing_range(slicing_input):
            range_input_err_alert = style_configs.activate_range_input_error_alert()
            return '', range_input_err_alert, dash.no_update

        else:
            range_list = slicing_input.split(':')
            if '%' in slicing_input:
                if not all('%' in idx for idx in range_list):
                    range_input_err_alert = style_configs.activate_range_input_error_alert()
                    return '', range_input_err_alert, dash.no_update

                else:
                    range_input_err_obj = _second_level_validity_check_on_user_input_range(range_list, df_features)
                    if range_input_err_obj != '':
                        return '', range_input_err_obj, dash.no_update

                    start_pct = int(range_list[0].replace('%', '')) / 100
                    start_idx = int(len(df_features) * start_pct)
                    stop_pct = int(range_list[1].replace('%', '')) / 100
                    stop_idx = int(len(df_features) * stop_pct)
                    return start_idx, stop_idx
            else:
                if range_list == ['']:
                    start_idx = default_start_idx
                    stop_idx = default_stop_idx
                    return start_idx, stop_idx
                else:
                    range_input_err_obj = _second_level_validity_check_on_user_input_range(range_list, df_features)
                    if range_input_err_obj != '':
                        return '', range_input_err_obj, dash.no_update

                    start_idx, stop_idx = int(range_list[0]), int(range_list[1])
                    return start_idx, stop_idx

    elif slicing_input is None or slicing_input == '':
        start_idx = default_start_idx
        stop_idx = default_stop_idx
        return start_idx, stop_idx


class FeatureDistribution:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.model_names = data_loader.get_model_list()
        self.is_bimodal = True if len(self.model_names) > 1 else False
        self.feature_to_exclude = None
        self.df_features = data_loader.get_features()
        self.specific_feature = []
        self.display_option = 'top'
        self.display_value = 3
        if self.analysis_type == 'regression':
            self.start_idx = int(len(self.df_features) * 0.8)
            self.stop_idx = len(self.df_features)
            self.range_selection_header = style_configs.DEFAULT_RANGE_SELECTION_TEXT_REG
        elif 'classification' in self.analysis_type:
            self.start_idx = None
            self.stop_idx = None
            self.range_selection_header = style_configs.DEFAULT_RANGE_SELECTION_TEXT_CLS
        self.fig_obj_dict = fig_plot_distribution_by_kl_div_ranking(self.data_loader,
                                                                    self.feature_to_exclude,
                                                                    self.start_idx,
                                                                    self.stop_idx,
                                                                    self.display_option,
                                                                    self.display_value)
        self.fig_objs_specific = fig_plot_distribution_by_specific_feature(self.data_loader,
                                                                            self.specific_feature,
                                                                            self.start_idx,
                                                                            self.stop_idx)

    def show(self):
        options_feature_ls_reg = [{'label': f'{col}', 'value': f'{col}'} for col in self.df_features.columns]
        slider_range_topN_bottomN = {n: str(n) for n in range(1, 11)}

        if self.analysis_type == 'regression' or (('classification' in self.analysis_type) and not detected_bimodal(self.model_names)):
            if isinstance(self.fig_obj_dict, list):
                self.fig_obj_dict = self.fig_obj_dict[0]
            if 'classification' in self.analysis_type:
                self.fig_objs_specific = self.fig_objs_specific[0]

            self.fig_objs_display_combined = _fig_objs_layout_based_on_display_option(self.fig_obj_dict, self.display_option)
            self.fig_objs_specific_feature_combined = _fig_objs_layout_based_on_specific_feature(self.specific_feature,
                                                                                                    self.fig_objs_specific)

        elif ('classification' in self.analysis_type) and detected_bimodal(self.model_names):
            self.fig_objs_display_combined = _model_pair_fig_objs_layout_on_display_option(self.fig_obj_dict, self.display_option)
            self.fig_objs_specific_feature_combined = _model_pair_fig_objs_layout_on_specific_feature(self.specific_feature,
                                                                                                        self.fig_objs_specific)

        distplot = dbc.Container([
                    dbc.Row(html.H5('Feature Distribution by KL-Divergence Ranking',
                                    id='title-after-topn-bottomn-reg',
                                    className='h5__cluster-section-title')),

                    # params selection section: feature_to_exclude, select topN/BottomN/Both, define slicing range
                    html.Div([
                        dbc.Row(html.Div(html.H6('Select feature to exclude ( if applicable ) :'), className='h6__feature-to-exclude')),
                        dbc.Col(dcc.Dropdown(id='select-feature-to-exclude-featdist',
                                            options=options_feature_ls_reg,
                                        value=[], multi=True)),
                        html.Br(),
                        dbc.Row([
                            dbc.Col([
                                dbc.Row(html.Div(
                                            html.H6('Display feature distribution by KL-Divergence score :'),
                                            className='h6__display-top-bottom-n')),
                                dbc.Row(dbc.FormGroup([
                                            dbc.RadioItems(
                                                options=[
                                                    {'label': 'Top N only', 'value': 'top'},
                                                    {'label': 'Bottom N only', 'value': 'bottom'},
                                                    {'label': 'Top N and Bottom N', 'value': 'both'}],
                                                value='top',
                                                id="select-radioitems-kldiv-top-or-bottom",
                                                inline=True, inputClassName='radiobutton__select-top-bottom-n', custom=False)])),
                                dcc.Slider(id='select-slider-top-bottom-range-featdist',
                                            min=1,
                                            max=_get_max_value_on_slider(self.df_features),  # max at 10
                                            step=1,
                                            value=3,
                                            marks=slider_range_topN_bottomN)], width=6),
                            dbc.Col([
                                dbc.Row(html.Div(html.H6(self.range_selection_header), className='h6__enter-data-range')),
                                dbc.Row(dbc.Input(id='input-range-to-slice-kldiv-featdist',
                                                    placeholder='start_index:stop_index',
                                                    type='text',
                                                    value=None)),
                                dbc.Row(html.Div(html.Pre(style_configs.input_range_subnote(self.df_features),
                                                            className='text__range-header-kldiv-featfist')))],
                                width=6)]),

                        html.Br(),
                        dbc.Row([
                            dbc.Col(width=10),
                            dbc.Col(dbc.Row(
                                        dcc.Loading(id='loading-output-kldiv-featdist',
                                                    type='circle', color='#a80202'),
                                    justify='right', className='loading__kldiv-featdist'), width=1),
                            dbc.Col(dbc.Row(
                                        dbc.Button("Update",
                                                    id='button-featdist-kldiv-update',
                                                    n_clicks=0,
                                                    className='button__update-dataset'),
                                    justify='right'))]),
                    ], className='border__select-dataset'),

                    html.Div(id='alert-range-input-error-kldiv-featdist'),
                    html.Div(self.fig_objs_display_combined, id='fig-objs-kl-div-by-ranking'),
                    dbc.Row(html.H5('Distribution of Specific Feature of Interest',
                                    id='title-specific-feat-dist-reg',
                                    className='h5__cluster-section-title')),

                    # params selection section: specific feature to inspect its distribution regardless of kl-div ranking
                    html.Div([
                        dbc.Row(html.Div(html.H6('Select specific feature of interest to inspect'),
                                        className='h6__cluster-instruction')),
                        dbc.Col(dcc.Dropdown(id='select-specific-feature-featdist',
                                            options=options_feature_ls_reg,
                                            value=[], multi=True)),
                        html.Br(),
                        dbc.Row([
                            dbc.Col([
                                dbc.Row(html.Div(html.H6(self.range_selection_header), className='h6__enter-data-range')),
                                dbc.Row(dbc.Input(id='input-range-to-slice-specific-feat-featdist',
                                                placeholder='start_index:stop_index',
                                                    type='text')),
                                dbc.Row(html.Div(html.Pre(style_configs.input_range_subnote(self.df_features),
                                                            className='text__range-header-kldiv-featfist')))],
                                width=6),
                            dbc.Col(width=4),
                            dbc.Col(dbc.Row(
                                        dcc.Loading(id='loading-output-specific-feat-featdist',
                                                    type='circle', color='#a80202'),
                                    justify='left', className='loading__specific-feat-featdist'), width=1),
                            dbc.Col(dbc.Row(
                                        dbc.Button("Update",
                                                    id='button-featdist-specific-feat-update',
                                                    n_clicks=0,
                                                    className='button__update-dataset'), justify='right'), )
                        ])
                    ], className='border__select-dataset'),
                    html.Div(id='alert-range-input-error-spedific-feat-featdist'),
                    html.Div(self.fig_objs_specific_feature_combined, id='fig-objs-kl-div-specific-feat'),
                    html.Br()], fluid=True)
        return distplot

    def callbacks(self):
        # callback on params related to top-n, botton-n / both
        @app.callback(
            Output('loading-output-kldiv-featdist', 'children'),
            Output('alert-range-input-error-kldiv-featdist', 'children'),
            Output('fig-objs-kl-div-by-ranking', 'children'),
            Input('button-featdist-kldiv-update', 'n_clicks'),
            State('select-feature-to-exclude-featdist', 'value'),
            State('select-radioitems-kldiv-top-or-bottom', 'value'),
            State('select-slider-top-bottom-range-featdist', 'value'),
            State('input-range-to-slice-kldiv-featdist', 'value'))
        def update_fig_based_on_topbtm_exclusion_idx_range_params(click_count, feature_to_exclude, display_option, display_value, slicing_input):
            if click_count > 0:
                range_input_err_alert = style_configs.no_error_alert()
                compact_outputs_kldiv = _generate_valid_start_stop_idx_based_on_usr_input_range(slicing_input,
                                                                                                self.df_features,
                                                                                                self.start_idx,
                                                                                                self.stop_idx)
                if len(compact_outputs_kldiv) == 2:
                    start_idx, stop_idx = compact_outputs_kldiv[0], compact_outputs_kldiv[1]

                    fig_obj_dict = fig_plot_distribution_by_kl_div_ranking(self.data_loader,
                                                                            feature_to_exclude,
                                                                            start_idx,
                                                                            stop_idx,
                                                                            display_option,
                                                                            display_value)
                    if self.analysis_type == 'regression' or \
                            (('classification' in self.analysis_type) and not detected_bimodal(self.model_names)):
                        if isinstance(fig_obj_dict, list):
                            fig_obj_dict = fig_obj_dict[0]
                        fig_objs_display_combined = _fig_objs_layout_based_on_display_option(fig_obj_dict, display_option)

                    elif ('classification' in self.analysis_type) and detected_bimodal(self.model_names):
                        fig_objs_display_combined = _model_pair_fig_objs_layout_on_display_option(fig_obj_dict, display_option)
                    return '', range_input_err_alert, fig_objs_display_combined
                else:
                    return compact_outputs_kldiv[0], compact_outputs_kldiv[1], compact_outputs_kldiv[2]
            else:
                raise PreventUpdate

        # callback on params related to specific feature chosen by user
        @app.callback(
            Output('loading-output-specific-feat-featdist', 'children'),
            Output('alert-range-input-error-spedific-feat-featdist', 'children'),
            Output('fig-objs-kl-div-specific-feat', 'children'),
            Input('button-featdist-specific-feat-update', 'n_clicks'),
            State('select-specific-feature-featdist', 'value'),
            State('input-range-to-slice-specific-feat-featdist', 'value'))
        def update_fig_based_on_specific_feat_idx_range_params(click_count, specific_feature, slicing_input):
            if click_count > 0:
                range_input_err_alert = style_configs.no_error_alert()
                compact_outputs_specific_feat = _generate_valid_start_stop_idx_based_on_usr_input_range(slicing_input,
                                                                                                self.df_features,
                                                                                                self.start_idx,
                                                                                                self.stop_idx)
                if len(compact_outputs_specific_feat) == 2:
                    start_idx, stop_idx = compact_outputs_specific_feat[0], compact_outputs_specific_feat[1]
                    fig_objs_specific = fig_plot_distribution_by_specific_feature(self.data_loader, specific_feature, start_idx, stop_idx)

                    if (self.analysis_type == 'regression') or \
                            (('classification' in self.analysis_type) and not detected_bimodal(self.model_names)):
                        if 'classification' in self.analysis_type:
                            fig_objs_specific = fig_objs_specific[0]
                        fig_objs_specific_feature_combined = _fig_objs_layout_based_on_specific_feature(specific_feature, fig_objs_specific)

                    elif ('classification' in self.analysis_type) and detected_bimodal(self.model_names):
                        fig_objs_specific_feature_combined = _model_pair_fig_objs_layout_on_specific_feature(specific_feature, fig_objs_specific)
                    return '', range_input_err_alert, fig_objs_specific_feature_combined
                else:
                    return compact_outputs_specific_feat[0], compact_outputs_specific_feat[1], compact_outputs_specific_feat[2]

            else:
                raise PreventUpdate

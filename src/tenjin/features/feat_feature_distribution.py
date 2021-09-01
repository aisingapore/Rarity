import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from tenjin.interpreters.structured_data import IntFeatureDistribution
from tenjin.visualizers.xfeature_distribution import plot_distribution_by_specific_feature, plot_distribution_by_kl_div_ranking


def fig_plot_distribution_by_kl_div_ranking(data_loader, feature_to_exclude, start_idx, stop_idx, display_option, display_value):
    kl_div_dict_sorted = IntFeatureDistribution(data_loader).xform(feature_to_exclude, start_idx, stop_idx)
    fig_obj_dict = plot_distribution_by_kl_div_ranking(kl_div_dict_sorted, display_option, display_value)
    return fig_obj_dict


def fig_plot_distribution_by_specific_feature(data_loader, specific_feature_ls, start_idx, stop_idx):
    kl_div_dict_sorted = IntFeatureDistribution(data_loader).xform(None, start_idx, stop_idx)
    fig_objs_specific = plot_distribution_by_specific_feature(specific_feature_ls, kl_div_dict_sorted)
    return fig_objs_specific


def _fig_objs_layout_based_on_display_option(fig_obj_dict, display_option):
    '''
    fig_obj_dict => already filtered to top n value selected by user
    '''
    fig_objs_display_combined = []

    if display_option == 'top':
        for n in range(len(fig_obj_dict[display_option])):
            top_n_fig_reg = fig_obj_dict[display_option][n][0]
            print(f'top_n_fig_reg: {top_n_fig_reg}')

            single_distplot_top = [html.H4(f'Top {n + 1}', className='h4__title-kl-div-rank-num-top'),
                                    dbc.Row([
                                        dbc.Row(dcc.Graph(id=f'fig-top{n + 1}-feat-dist-reg', figure=top_n_fig_reg), justify='center')],
                                        justify='center', className='border__kl-div-figs')]
            fig_objs_display_combined += single_distplot_top

    elif display_option == 'bottom':
        for n in range(len(fig_obj_dict[display_option])):
            btm_n_fig_reg = fig_obj_dict[display_option][-(n + 1)][0]  # n + 1 => to avoid -0
            print(f'btm_n_fig_reg: {btm_n_fig_reg}')

            single_distplot_btm = [html.H4(f'Bottom {n + 1}', className='h4__title-kl-div-rank-num-bottom'),
                                    dbc.Row([
                                        dbc.Row(dcc.Graph(id=f'fig-bottom{n + 1}-feat-dist-reg', figure=btm_n_fig_reg), justify='center')],
                                        justify='center', className='border__kl-div-figs')]
            fig_objs_display_combined += single_distplot_btm

    elif display_option == 'both':
        for n in range(len(fig_obj_dict['top'])):
            top_n_fig_reg = fig_obj_dict['top'][n][0]
            btm_n_fig_reg = fig_obj_dict['bottom'][-(n + 1)][0]

            single_distplot_top_btm_pair = [dbc.Row([
                                                dbc.Col([
                                                    html.H4(f'Top {n + 1}', className='h4__title-kl-div-rank-num-top'),
                                                    dbc.Row([
                                                        dbc.Col(dcc.Graph(id=f'fig-top{n + 1}-feat-dist-reg', figure=top_n_fig_reg))],
                                                        justify='center', className='border__kl-div-figs-both-left')], width=6),

                                                dbc.Col([
                                                    html.H4(f'Bottom {n + 1}', className='h4__title-kl-div-rank-num-bottom'),
                                                    dbc.Row([
                                                        dbc.Col(dcc.Graph(id=f'fig-bottom{n + 1}-feat-dist-reg', figure=btm_n_fig_reg))],
                                                        justify='center', className='border__kl-div-figs-both-right')], width=6)])]

            fig_objs_display_combined += single_distplot_top_btm_pair
    return fig_objs_display_combined


def _fig_objs_layout_based_on_specific_feature(specific_feature, fig_objs_specific):
    fig_objs_specific_feature_combined = []
    if len(specific_feature) > 0:
        for i, specific_fig in enumerate(fig_objs_specific):
            single_displot_specific = [dbc.Row(dcc.Graph(id=f'fig-specific{i}-feat-dist-reg', figure=specific_fig), 
                                                justify='center', className='border__specific-feat-dist')]
            fig_objs_specific_feature_combined += single_displot_specific
    return fig_objs_specific_feature_combined


class FeatureDistribution:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.model_names = data_loader.get_model_list()
        self.is_bimodal = True if len(self.model_names) > 1 else False
        self.feature_to_exclude = None

        if self.analysis_type == 'regression':
            self.df_features = data_loader.get_features()
            self.specific_feature = [self.df_features.columns[0]]
            self.start_idx = int(len(self.df_features) * 0.8)
            self.stop_idx = len(self.df_features)
            self.display_option = 'top'
            self.display_value = 3
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
        if self.analysis_type == 'regression':
            options_feature_ls_reg = [{'label': f'{col}', 'value': f'{col}'} for col in self.df_features.columns]
            slider_range_topN_bottomN = {n: str(n) for n in range(1, 11)}

            self.fig_objs_display_combined = _fig_objs_layout_based_on_display_option(self.fig_obj_dict, self.display_option)
            self.fig_objs_specific_feature_combined = _fig_objs_layout_based_on_specific_feature(self.specific_feature, self.fig_objs_specific)

            fig_objs_specific_feature_combined = []
            if len(self.specific_feature) > 0:
                for i, specific_fig in enumerate(self.fig_objs_specific):
                    single_displot_specific = [dbc.Row(
                                                    dcc.Graph(id=f'fig-specific{i}-feat-dist-reg', figure=specific_fig), 
                                                    justify='center', className='border__optimum-cluster')]
                    fig_objs_specific_feature_combined += single_displot_specific

            distplot = dbc.Container([
                        dbc.Row(html.H5('Feature Distribution by KL-Divergence Ranking',
                                        id='title-after-topn-bottomn-reg',
                                        className='h5__cluster-section-title')),

                        # params selection section: feature_to_exclude, select topN/BottomN/Both, define slicing range
                        html.Div([
                            dbc.Row(html.Div(html.H6('Select feature to exclude ( if applicable ) :'), className='h6__feature-to-exclude')),
                            dbc.Col(dcc.Dropdown(id='select-feature-to-exclude-reg',
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
                                                    id="select-radioitems-dist-top-or-bottom",
                                                    inline=True, inputClassName='radiobutton__select-top-bottom-n', custom=False)])),
                                    dcc.Slider(id='select-slider-top-n-reg', min=1, max=10, step=1, value=3,
                                                marks=slider_range_topN_bottomN)], width=6),
                                dbc.Col([
                                    dbc.Row(html.Div(
                                                html.H6("Enter range of data to compare distribution " \
                                                        "( default slicing - last 20% of dataset) :"), className='h6__enter-data-range')),
                                    dbc.Row(dbc.Input(id="input-range-to-slice-reg", placeholder="start_index:stop_index", type="text"))],
                                    width=6)]),

                            html.Br(),
                            dbc.Row([
                                dbc.Col(dbc.Row(
                                            dcc.Loading(id='loading-output-misspred-dataset-cls',
                                                        type='circle', color='#a80202'),
                                        justify='left', className='loading__loss-cluster'), width=1),
                                dbc.Col(dbc.Row(
                                            dbc.Button("Update",
                                                        id='button-misspred-dataset-update-cls',
                                                        n_clicks=0,
                                                        className='button__update-dataset'),
                                        justify='right'))]),
                        ], className='border__select-dataset'),

                        html.Div(self.fig_objs_display_combined),
                        dbc.Row(html.H5('Distribution of Specific Feature of Interest',
                                        id='title-specific-feat-dist-reg',
                                        className='h5__cluster-section-title')),

                        # params selection section: specific feature to inspect its distribution regardless of kl-div ranking
                        html.Div([
                            dbc.Row(html.Div(html.H6('Select specific feature of interest to inspect'),
                                            className='h6__cluster-instruction')),
                            dbc.Col(dcc.Dropdown(id='select-specific-feature-to-inspect-reg',
                                                options=options_feature_ls_reg,
                                                value=[options_feature_ls_reg[0]], multi=True)),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Row(html.Div(html.H6('Enter range of data to compare distribution ' \
                                                            '( default slicing - last 20% of dataset) :'),
                                                            className='h6__enter-data-range')),
                                    dbc.Row(dbc.Input(id="input-range-to-slice-reg", placeholder="start_index:stop_index", type="text"))],
                                    width=6),
                                dbc.Col(dbc.Row(
                                            dcc.Loading(id='loading-output-misspred-dataset-cls',
                                                        type='circle', color='#a80202'),
                                        justify='left', className='loading__loss-cluster'), width=1),
                                dbc.Col(dbc.Row(
                                                dbc.Button("Update",
                                                            id='button-misspred-dataset-update-cls',
                                                            n_clicks=0,
                                                            className='button__update-dataset'), justify='right'))
                            ])
                        ], className='border__select-dataset'),
                        html.Div(self.fig_objs_specific_feature_combined),
                        html.Br()], fluid=True)
        return distplot

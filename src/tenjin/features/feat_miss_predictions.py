from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from tenjin.app import app
from tenjin.interpreters.structured_data import IntMissPredictions
from tenjin.visualizers import miss_predictions as viz_misspred

INSTRUCTION_TEXT = 'Click and drag on the graph to select the range of data points to inspect feature values'


def fig_probabilities_spread_pattern(data_loader):
    """
    Create confusion matrix

    Arguments:
        data_loader {class object} 
        -- output from data loader pipeline

    Returns:
        plotly graph 
        -- displaying confusion matrix details
    """
    _, ls_dfs_by_label, ls_dfs_by_label_state = IntMissPredictions(data_loader).xform()

    fig_objs_all_models = []
    tables_all_models = []
    for i, ls_dfs in enumerate(ls_dfs_by_label):
        fig_objs_per_model = []
        tables_per_model = []
        for j, df_specific_label in enumerate(ls_dfs):
            fig_obj = viz_misspred.plot_probabilities_spread_pattern(df_specific_label)
            fig_objs_per_model.append(fig_obj)

            table_j = viz_misspred.plot_simple_probs_spread_overview(ls_dfs_by_label_state[i][j])
            tables_per_model.append(table_j)
        fig_objs_all_models.append(fig_objs_per_model)
        tables_all_models.append(tables_per_model)

    return fig_objs_all_models, tables_all_models


def fig_plot_prediction_offset_overview(data_loader):
    df = IntMissPredictions(data_loader).xform()
    fig_obj = viz_misspred.plot_prediction_offset_overview(df)
    return fig_obj, df


def table_to_filtered_datapoints_reg(table_id, customized_cols):
    tab_obj = viz_misspred.reponsive_table_to_filtered_datapoints_reg(table_id, customized_cols)
    return tab_obj


def convert_filtered_data_to_df(relayout_data, df, models):
    """convert raw data format from relayout selection range by user into the correct df fit for viz purpose

    Arguments:
        relayout_data {dict}: data containing selection range indices returned from plotly graph
        df {pandas dataframe}: dataframe tap-out from interpreters pipeline
        models {list}: model names defined by user during spin-up of Tenjin app

    Returns:
        pandas dataframe 
        -- dataframe fit for the responsive table-graph filtering
    """
    try:
        x_start_idx = int(relayout_data['xaxis.range[0]']) if relayout_data['xaxis.range[0]'] > 0 else 0
        x_stop_idx = int(relayout_data['xaxis.range[1]']) if relayout_data['xaxis.range[1]'] < len(df) else len(df) - 1
        y_start_idx = int(relayout_data['yaxis.range[0]'])
        y_stop_idx = int(relayout_data['yaxis.range[1]'])
        offset_cols = [col for col in df.columns if 'offset_' in col]

        df_filtered = df.iloc[df.index[x_start_idx]:df.index[x_stop_idx]]

        if len(offset_cols) == 1:
            if offset_cols[0].replace('offset_', '') == models[0]:
                df_final = df_filtered[(df_filtered[offset_cols[0]] >= y_start_idx) & (df_filtered[offset_cols[0]] <= y_stop_idx)]
            else:
                df_final = df_filtered[(df_filtered[offset_cols[1]] >= y_start_idx) & (df_filtered[offset_cols[1]] <= y_stop_idx)]

        elif len(offset_cols) > 1:
            df_final_m1 = df_filtered[(df_filtered[offset_cols[0]] >= y_start_idx) & (df_filtered[offset_cols[0]] <= y_stop_idx)]
            df_final_m2 = df_filtered[(df_filtered[offset_cols[1]] >= y_start_idx) & (df_filtered[offset_cols[1]] <= y_stop_idx)]

            final_filtered_idx = set(df_final_m1.index).union(set(df_final_m2.index))
            df_final = df_filtered[df_filtered['index'].isin(list(final_filtered_idx))]
        return df_final

    except TypeError:  # during first startup when no filtered data applied
        pass


class MissPredictions:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.is_bimodal = True if len(self.data_loader.get_model_list()) > 1 else False

        if self.analysis_type == 'regression':
            self.preds_offset, self.df = fig_plot_prediction_offset_overview(self.data_loader)
            self.cols_dash_table = [col.replace('_', ' ') for col in self.df.columns]
            self.table_obj_reg = table_to_filtered_datapoints_reg('responsive-dash-table', self.cols_dash_table)

        elif 'classification' in self.analysis_type:
            self.probs_pattern, self.label_state = fig_probabilities_spread_pattern(self.data_loader)

    def show(self):
        if self.analysis_type == 'regression':
            miss_preds = dbc.Container([
                                dbc.Row(dcc.Graph(id='filter-datapoint-reg', 
                                                figure=self.preds_offset,), 
                                                justify="center", 
                                                className="border__common-misspred-reg"),
                                html.Div(html.H6(INSTRUCTION_TEXT), className="h6__dash-table-note"),
                                html.Div(self.table_obj_reg, className="div__table-proba-spread"),
                                html.Br()
                        ], fluid=True)

        elif 'classification' in self.analysis_type:
            fig_objs_model_1 = self.probs_pattern[0]
            tables_model_1 = self.label_state[0]

            if self.is_bimodal and 'classification' in self.analysis_type:  # cover bimodal_binary and bimodal_multiclass
                fig_objs_model_2 = self.probs_pattern[1]
                tables_model_2 = self.label_state[1]

                dash_fig_ls = []
                for i in range(0, len(fig_objs_model_1), 2):
                    try:  # enabling the display of a pair of figures for better comparison view
                        fig_pair = dbc.Row([
                                        dbc.Col([dbc.Row([
                                                dbc.Col([
                                                    dbc.Row(dcc.Graph(figure=fig_objs_model_1[i]), justify="center"),
                                                    dbc.Row(html.Div(
                                                        html.Div(tables_model_1[i], className="div__table-proba-spread")), 
                                                        justify="center"),
                                                ]),
                                                dbc.Col([
                                                    dbc.Row(dcc.Graph(figure=fig_objs_model_2[i]), justify="center"),
                                                    dbc.Row(html.Div(
                                                        html.Div(tables_model_2[i], className="div__table-proba-spread")), 
                                                        justify="center"),
                                                ]),
                                                ])
                                        ], className="border__common"),

                                        dbc.Col([dbc.Row([
                                                dbc.Col([
                                                    dbc.Row(dcc.Graph(figure=fig_objs_model_1[i + 1]), justify="center"),
                                                    dbc.Row(html.Div(
                                                        html.Div(tables_model_1[i + 1], className="div__table-proba-spread")), 
                                                        justify="center"),
                                                ]),
                                                dbc.Col([
                                                    dbc.Row(dcc.Graph(figure=fig_objs_model_2[i + 1])),
                                                    dbc.Row(html.Div(
                                                        html.Div(tables_model_2[i + 1], className="div__table-proba-spread")), 
                                                        justify="center"),
                                                ]),
                                                ])
                                        ], className="border__common")
                                    ])
                        dash_fig_ls.append(fig_pair)

                    except IndexError:  # handling the last odd figure that can't be paired out
                        fig_pair = dbc.Row([
                                        dbc.Col([dbc.Row([
                                                dbc.Col([
                                                    dbc.Row(dcc.Graph(figure=fig_objs_model_1[i]), justify="center"),
                                                    dbc.Row(html.Div(
                                                        html.Div(tables_model_1[i], className="div__table-proba-spread")), 
                                                        justify="center"),
                                                ]),
                                                dbc.Col([
                                                    dbc.Row(dcc.Graph(figure=fig_objs_model_2[i]), justify="center"),
                                                    dbc.Row(html.Div(
                                                        html.Div(tables_model_2[i], className="div__table-proba-spread")), 
                                                        justify="center"),
                                                ]),
                                                ])
                                        ], className="border__common")])
                        dash_fig_ls.append(fig_pair)

                miss_preds = dbc.Container(dash_fig_ls, fluid=True)

            elif not self.is_bimodal and 'binary' in self.analysis_type:  # single modal binary classification
                miss_preds = dbc.Container([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Row([dcc.Graph(figure=fig_objs_model_1[0])], justify="center"),
                                                    dbc.Row([
                                                        html.Div(html.Div(tables_model_1[0], className="div__table-proba-spread")),
                                                    ], justify="center"),
                                                ]),

                                                dbc.Col([
                                                    dbc.Row([dcc.Graph(figure=fig_objs_model_1[1])], justify="center"),
                                                    dbc.Row([
                                                        html.Div(html.Div(tables_model_1[1], className="div__table-proba-spread")),
                                                    ], justify="center"),
                                                ]),
                                            ])

                                        ], className="border__common"),

                                        dbc.Col([], className="border__common")
                                    ]),

                                    dbc.Row([]),
                            ], fluid=True)

            elif not self.is_bimodal and 'multiclass' in self.analysis_type:  # single modal multi-class classification
                dash_fig_ls = []
                for i in range(0, len(fig_objs_model_1)):
                    fig_pair = dbc.Col([
                                    dbc.Row([dcc.Graph(figure=fig_objs_model_1[i])], justify="center"),
                                    dbc.Row([
                                        html.Div(html.Div(tables_model_1[i], className="div__table-proba-spread")),
                                    ], justify="center"),
                                ], className="border__common")
                    dash_fig_ls.append(fig_pair)

                miss_preds = dbc.Container([
                                    dbc.Row(dbc.Col([dbc.Row(dash_fig_ls)])), 
                                    dbc.Row(),
                            ], fluid=True)
        return miss_preds

    def callback(self):
        @app.callback(
            Output('responsive-dash-table', 'data'), 
            Input('filter-datapoint-reg', 'relayoutData'))
        def display_relayout_data(relayoutData):
            if relayoutData is not None:
                try:
                    self.df = self.df.round(2)
                    print(f'len_df_b4_dash_table: {len(self.df)}')
                except TypeError:
                    self.df

                models = self.data_loader.get_model_list()
                df_final = convert_filtered_data_to_df(relayoutData, self.df, models)
                df_final.columns = self.cols_dash_table
                data_relayout = df_final.to_dict('records')
                return data_relayout
            else:
                raise PreventUpdate

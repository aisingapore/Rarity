import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from ..interpreters.structured_data import IntMissPredictions
from ..visualizers import miss_predictions as viz_misspred


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


class MissPredictions:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.is_bimodal = True if len(self.data_loader.get_model_list()) > 1 else False

        if self.analysis_type == 'regression':
            pass
        elif 'classification' in self.analysis_type:
            self.probs_pattern, self.label_state = fig_probabilities_spread_pattern(self.data_loader)

    def show(self):
        if self.analysis_type == 'regression':
            miss_preds = dbc.Container([
                                dbc.Row([
                                    dbc.Col([
                                        # dcc.Graph(figure=self.pred_actual),
                                        dbc.Row(dcc.Graph(figure=self.pred_offset), justify="center")
                                    ], className="border__common"), 
                                ]), 

                                html.Br(),
                        ], fluid=True)
        elif 'classification' in self.analysis_type:
            fig_objs_model_1 = self.probs_pattern[0]
            tables_model_1 = self.label_state[0]
            if self.is_bimodal and 'classification' in self.analysis_type:  # cover bimodal_binary and bimodal_multiclass
                # if self.is_bimodal and 'multiclass' in self.analysis_type:
                fig_objs_model_2 = self.probs_pattern[1]
                tables_model_2 = self.label_state[1]

                dash_fig_ls = []
                for i in range(0, len(fig_objs_model_1), 2):
                    try:
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
                                        ], className="border__common")])

                        dash_fig_ls.append(fig_pair)
                    except IndexError:
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
                                    ])
                        dash_fig_ls.append(fig_pair)

                miss_preds = dbc.Container(dash_fig_ls, fluid=True)

            elif not self.is_bimodal and 'binary' in self.analysis_type:
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

            elif not self.is_bimodal and 'multiclass' in self.analysis_type:
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

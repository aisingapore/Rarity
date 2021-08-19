import dash_table


def reponsive_table_to_filtered_datapoints(data, customized_cols, header, exp_format):
    table_obj = dash_table.DataTable(
                    data=data,
                    columns=[{'id': c, 'name': c} for c in customized_cols],
                    page_action='none',
                    fixed_rows={'headers': True},
                    # fixed_columns={'headers': True, 'data': 1},  # alighment will be out
                    # - bug reported in dash repo
                    style_data={'whiteSpace': 'normal', 'height': 'auto'},
                    style_cell={'textAlign': 'center',
                                'border': '1px solid rgb(229, 211, 197)',
                                'font-family': 'Arial',
                                'margin-bottom': '0',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'minWidth': '200px',
                                'width': '200px',
                                'maxWidth': '200px'},
                    style_header=header,
                    style_table={'width': 1400,
                                'height': 200,
                                'margin': 'auto'},
                    export_format=exp_format),
    return table_obj

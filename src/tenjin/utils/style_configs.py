import dash_bootstrap_components as dbc


INSTRUCTION_TEXT_SHARED = 'Click and drag on the graph to select the range of data points to inspect feature values.'
INSTRUCTION_TEXT_REG = 'To reset back to default settings, hover over icons on the top right of the graph and click "Autoscale" icon.'
WARNING_TEXT = 'To inspect new range of datapoints in different graph, please first reset the earlier selection by clicking "Autoscale" icon ' \
                'at the top right corner of the graph.'

DEFAULT_HEADER_STYLE = {'fontWeight': 'bold', 'color': 'white', 'backgroundColor': '#7e746d', 'border': '1px solid rgb(229, 211, 197)'}
DEFAULT_TITLE_STYLE = {'visibility': 'visible'}
DEFAULT_PLOT_NAME_STYLE = {'visibility': 'visible'}

DEFAULT_RANGE_SELECTION_TEXT_REG = "Enter range of data to compare distribution ( default slicing - last 20% of dataset) :"
DEFAULT_RANGE_SELECTION_TEXT_CLS = "Enter range of data to compare distribution ( default slicing - full range ):"


def default_header_style():
    DEFAULT_HEADER_STYLE['visibility'] = 'visible'
    return DEFAULT_HEADER_STYLE


def hidden_title_style():
    DEFAULT_TITLE_STYLE['visibility'] = 'hidden'
    return DEFAULT_TITLE_STYLE


def hidden_plot_name_style():
    DEFAULT_PLOT_NAME_STYLE['visibility'] = 'hidden'
    return DEFAULT_PLOT_NAME_STYLE


def collapse_header_style():
    DEFAULT_HEADER_STYLE['border'] = 'none'
    DEFAULT_HEADER_STYLE['visibility'] = 'collapse'
    return DEFAULT_HEADER_STYLE


def dummy_alert():
    alert_obj = dbc.Alert(color="light", style={'visibility': 'hidden'})
    return alert_obj


def activate_alert():
    alert_obj = dbc.Alert(INSTRUCTION_TEXT_REG, color="secondary", className='alert__note-reg')
    return alert_obj


def activate_cluster_error_alert(label_class):
    err_message = f'Miss prediction data points are not sufficient for auto-clustering in < Class {label_class} >. ' \
                    'Minimum number of datapoint for auto-clustering is 8 datapoints per class per model'
    alert_obj = dbc.Alert(err_message, color='warning', dismissable=True, is_open=True)
    return alert_obj


def no_cluster_error_alert():
    alert_obj = dbc.Alert('', color='warning', dismissable=True, is_open=False)
    return alert_obj
